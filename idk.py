"""
SOLVER.py — PyFluent airfoil solver
AoA sweep in a single Fluent session with automatic steady → transient switching.
Decision logic follows LOGIC.md (Re 60k–500k, low-Re airfoil validation).

Engineering context
-------------------
This script is designed for low-Re airfoil validation against UIUC wind tunnel data
(Selig et al., "Summary of Low-Speed Airfoil Data", UIUC Aerodynamics Group).

The key challenge at Re 60k–500k is the laminar separation bubble (LSB): a thin region
where the BL separates laminar, transitions turbulent, and reattaches. RANS cannot
resolve the LSB directly — the choice of turbulence model determines whether the
transition is modelled at all, which dominates CD error at these Reynolds numbers.

Architecture decision — single Fluent session:
  All AoA cases for one (Re, model) combination run inside one launch_fluent() call.
  Mesh is read once; only BCs, monitors, and time-mode change between cases.
  This avoids the ~90-second Fluent startup cost per AoA and keeps GPU/CPU memory warm.
  The trade-off: if Fluent crashes mid-sweep, the session must be restarted from scratch.
  Mitigation: .cas.h5 is saved after every AoA case.
"""

import math
import time
import numpy as np
import psutil
from pathlib import Path
from typing import Any
import ansys.fluent.core as pyfluent  # type: ignore

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
NU_AIR  = 1.48e-5   # kinematic viscosity [m²/s] (air, ~20°C)
RHO_AIR = 1.225     # density [kg/m³]


class FluentAirfoilSolver:
    """
    Run an AoA sweep for one airfoil / Re combination in a single Fluent session.

    Steady RANS is attempted first; if CL oscillations exceed thresholds
    (LOGIC.md Sec 2.4), the solver switches to transient in the SAME session —
    no need to re-launch Fluent or re-set boundary conditions.

    Usage
    -----
    solver = FluentAirfoilSolver(
        mesh_path    = r"data/runs/NACA6409/meshes/Fine_NACA6409_2d.msh.h5",
        airfoil_name = "NACA6409",
        chord_m      = 1.000,
        Re           = 61_000,
        AoA_list     = [0.26, 5.00],
        turb_model   = "k-kl-w",
        output_dir   = r"data/runs/NACA6409/results/Re61k_kkl",
        animate      = True,   # record Cp/Cf/ψ AVI during transient runs
    )
    results = solver.run_aoa_sweep()
    """

    def __init__(self, mesh_path, airfoil_name, chord_m, Re, AoA_list,
                 turb_model, output_dir, processor_count=8, animate=False, debug=True):
        self.mesh_path      = Path(mesh_path)
        self.airfoil_name   = airfoil_name
        self.chord_m        = chord_m
        self.Re             = Re
        self.AoA_list       = AoA_list
        self.turb_model     = turb_model  # 'k-kl-w' | 'transition-sst' | 'k-omega-sst'
        self.output_dir     = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processor_count = processor_count
        self.animate         = animate          # save Cp animation during transient runs
        self.debug           = debug            # True → _confirm() prompts; False → auto-proceed

        self.V_inf  = Re * NU_AIR / chord_m
        self.T_conv = chord_m / self.V_inf

        # Washout and averaging multipliers from LOGIC.md Sec 2.7
        if Re < 100_000:
            self.washout_mult = 10
            self.avg_mult     = 10
        elif Re < 200_000:
            self.washout_mult = 7
            self.avg_mult     = 5
        else:
            self.washout_mult = 5
            self.avg_mult     = 5

        self.session: Any      = None
        self.s: Any            = None   # alias for session.settings
        self.airfoil_zones     = []     # populated after mesh load
        self._postproc_ready   = False  # set after first hybrid_initialize

    # -------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------

    def _compute_flow_params(self, AoA_deg):
        """
        Velocity components, force direction vectors, time step for given AoA.

        Force vectors must update with every AoA — this is a common source of error.
        Fluent's force monitors integrate pressure and shear on the selected zones,
        then project onto the provided unit vector.  If you keep the vectors fixed at
        AoA=0, you get X-force and Y-force, NOT lift and drag.

          lift = F · (-sin α, cos α)   ← perpendicular to freestream
          drag = F · ( cos α, sin α)   ← parallel to freestream

        Reference values (density, velocity, area) are set once in _setup_session
        and do NOT change with AoA — Fluent uses them only to non-dimensionalise,
        and V_inf / ρ_ref do not vary across the sweep.
        """
        a = math.radians(AoA_deg)
        return {
            'Vx':       self.V_inf * math.cos(a),
            'Vy':       self.V_inf * math.sin(a),
            'lift_vec': [-math.sin(a), math.cos(a)],   # perpendicular to freestream
            'drag_vec': [ math.cos(a), math.sin(a)],   # parallel to freestream
            'dt':       5e-4 * self.T_conv,
        }

    def _confirm(self, message):
        """Block and ask for user confirmation. Raises RuntimeError on 'n'.
        Skipped entirely when debug=False.
        """
        if not self.debug:
            print(f"  [auto] {message}")
            return
        print(f"\n[CONFIRM] {message}")
        resp = input("  Proceed? (y/n): ").strip().lower()
        if resp != 'y':
            raise RuntimeError(f"User aborted at: {message}")
        print("  → OK")

    def _get_airfoil_zones(self):
        """Collect all wall zones whose name contains 'airfoil'."""
        all_walls = self.s.setup.boundary_conditions.wall.get_object_names()
        zones = [z for z in all_walls if 'airfoil' in z.lower()]
        if not zones:
            raise RuntimeError(
                f"No airfoil wall zones found. All walls: {all_walls}"
            )
        print(f"  Airfoil zones found: {zones}")
        return zones

    def _boost_priority(self):
        """Set HIGH_PRIORITY_CLASS on all fl_mpi.exe / fluent.exe processes.

        Requires the script to be run as Administrator — silently skips
        any process it doesn't have permission to modify.
        """
        targets = {'fl_mpi.exe', 'fluent.exe'}
        boosted = 0
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] and proc.info['name'].lower() in targets:
                try:
                    proc.nice(psutil.HIGH_PRIORITY_CLASS)
                    boosted += 1
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
        if boosted:
            print(f"  [solver] HIGH_PRIORITY_CLASS set on {boosted} Fluent process(es)")
        else:
            print("  [solver] priority boost skipped (run as Administrator to enable)")

    # -------------------------------------------------------------------
    # SESSION SETUP
    # -------------------------------------------------------------------

    def _setup_session(self):
        """
        Launch Fluent solver, read mesh, set reference values.

        Reference area convention (2D):
          area = chord × 1.0 m  — the "1 m" is a nominal span depth.
          Fluent integrates per-unit-span pressure and shear in 2D, so the
          dimensional force is [N/m].  Dividing by (½ρV²·c·1) gives the
          correct dimensionless CL/CD per unit span.
          Do NOT set area = chord² or use actual wing area here.

        The mesh confirmation prompt is intentional: a corrupted mesh import,
        wrong unit scaling, or misnamed zones will waste hours of compute.
        Check: geometry looks correct in GUI, named selections visible, no
        negative-volume cells in the mesh quality report.
        """
        print(f"\n[LAUNCH] Fluent — Re={self.Re}, model={self.turb_model}, "
              f"V∞={self.V_inf:.2f} m/s")

        self._t0 = time.time()
        self.session = pyfluent.launch_fluent(
            product_version  = "25.2.0",
            mode             = "solver",
            dimension        = 2,
            precision        = "double",
            processor_count  = self.processor_count,
            ui_mode          = "gui",         # change to "no_gui" for headless runs
            cleanup_on_exit  = False,         # keep Fluent open if Python crashes/exits
            # graphics_driver = "dx11",
        )
        self.s = self.session.settings
        self._boost_priority()

        # Picture settings — set once
        self.s.results.graphics.picture = {
            'x_resolution':       1920,
            'y_resolution':       1080,
            'invert_background':  False,
            'use_window_resolution': False,
        }

        # Read mesh
        print(f"  Reading mesh: {self.mesh_path}")
        self.s.file.read(file_name=str(self.mesh_path), file_type="mesh")

        self._confirm(
            "Mesh loaded — check in GUI: geometry, zone names, no bad cells."
        )

        self.airfoil_zones = self._get_airfoil_zones()

        # Pressure-based solver
        self.s.setup.general.solver.type = "pressure-based"

        # Reference values — set ONCE, do not change with AoA (LOGIC.md Sec 5)
        self.s.setup.reference_values.velocity = self.V_inf
        self.s.setup.reference_values.density  = RHO_AIR
        self.s.setup.reference_values.length   = self.chord_m
        self.s.setup.reference_values.area     = self.chord_m * 1.0  # 2D: chord × 1 m span

    # -------------------------------------------------------------------
    # TURBULENCE MODEL
    # -------------------------------------------------------------------

    def _set_turbulence_model(self):
        """
        Set viscous model. Called once after mesh load.

        Model selection rationale (LOGIC.md Sec 2.2):

          k-kl-w (Walters & Cokljat, 2008) — Re < 100k
            Three equations: turbulent KE (kt), laminar KE (kl), specific dissipation (ω).
            Bypasses the γ-Reθ transition correlation entirely; instead, kl accumulates
            in the pre-transitional BL and is converted to kt when local turbulence
            production thresholds are exceeded.  Better suited to natural transition at
            very low Re where the Langtry-Menter correlations were not calibrated.
            Residual equations to monitor: continuity, x-vel, y-vel, kl, kt, omega.

          transition-sst (Langtry-Menter γ-Reθ, 2009) — Re 100k–500k
            Four equations: k, ω (SST base) + γ (intermittency) + Reθ_t (transition momentum thickness).
            γ drives the BL from laminar to turbulent at the predicted transition point.
            γ_eff (effective intermittency) is the clipped value that enters production terms —
            the γ_eff contour directly shows the active transition zone.

          k-omega-sst — baseline comparison
            Fully turbulent: assumes turbulent from the leading edge.
            Counter-intuitively gives LOWER total CD than transition models at Re 60k–200k —
            not because turbulent skin friction is lower (it isn't: Cf,turb ∝ Re^(-1/5) vs
            Cf,lam ∝ Re^(-1/2)), but because at these Re the dominant drag source is LSB
            *pressure* drag. A fully turbulent BL never separates to form a bubble, so the
            large pressure-drag penalty from the laminar separation bubble is absent entirely.
            Result: artificially low CD, wrong physics — use only as lower-bound reference.
            Useful as upper-bound comparison; do not use as primary model for low-Re validation.
            Note: Fluent's k-omega model defaults to SST subtype — no additional setting needed.
        """
        if self.turb_model == 'k-kl-w':
            self.s.setup.models.viscous.model = 'k-kl-w'

        elif self.turb_model == 'transition-sst':
            self.s.setup.models.viscous.model = 'transition-sst'

        elif self.turb_model == 'k-omega-sst':
            self.s.setup.models.viscous.model = 'k-omega'  # defaults to SST in Fluent

        else:
            raise ValueError(f"Unknown turbulence model: {self.turb_model}")

        print(f"  Turbulence model set: {self.turb_model}")

    # -------------------------------------------------------------------
    # BOUNDARY CONDITIONS
    # -------------------------------------------------------------------

    def _set_boundary_conditions(self, AoA_deg):
        """
        Set velocity components and turbulence BCs for all inlet zones.
        UIUC tunnel turbulence (LOGIC.md Sec 2.8): TI=0.08%, TVR=2.

        Turbulence intensity choice:
          UIUC low-speed tunnel TI < 0.1% (very clean facility).
          Using a higher TI (e.g. 1%, Fluent default) artificially trips the BL
          earlier, shrinks the LSB, reduces CD, and makes the model look better
          than it is.  Matching the facility TI is required for honest validation.
          TVR=2 is a low value (essentially laminar freestream turbulence level).

        Velocity decomposition:
          The domain is fixed; AoA is applied by rotating the velocity vector.
          Vx = V·cos(α), Vy = V·sin(α) on a horizontal airfoil.
          This is equivalent to rotating the airfoil in a real tunnel — the
          pressure field and wake angle both respond correctly.

        Pressure outlet:
          Default gauge pressure = 0 is correct for incompressible external aero.
          It sets the reference pressure level, not an absolute value.
          Backflow TI/TVR at the outlet only matters if flow re-enters the domain,
          which should not happen for a well-placed outlet (>10c downstream).
        """
        p = self._compute_flow_params(AoA_deg)

        turb_bc = {
            'turbulent_intensity':       0.008,  # % — Fluent takes percentage
            'turbulent_viscosity_ratio': 2,
        }

        vel_bc = {
            'momentum': {
                'velocity_specification_method': 'Components',
                'velocity_components': [p['Vx'], p['Vy']],
            },
            'turbulence': turb_bc,
        }

        for zone in ['inlet', 'inlet_top', 'inlet_bot']:
            self.s.setup.boundary_conditions.velocity_inlet[zone] = vel_bc

        # Outlet: default gauge pressure = 0 is correct — no need to set it.
        # Backflow turbulence is also default (TI=5%, TVR=10) which is acceptable
        # since backflow only matters if the outlet is inside the domain (it shouldn't be).

        print(
            f"  BCs set: V∞={self.V_inf:.6f} m/s | "
            f"Vx={p['Vx']:.6f} m/s, Vy={p['Vy']:.6f} m/s | "
            f"|V|={math.hypot(p['Vx'], p['Vy']):.6f} m/s"
        )

    # -------------------------------------------------------------------
    # SOLVER METHODS — STEADY
    # -------------------------------------------------------------------

    def _set_methods_steady(self, conservative=False):
        """
        Coupled + Second Order Upwind.
        Standard URFs (LOGIC.md Sec 2.5 standard set) or conservative set.
        NOTE: Fluent resets to SIMPLE when switching time modes — always call
              this after any time-mode change.

        Why Coupled over SIMPLE/SIMPLEC:
          SIMPLE pressure-correction can stall on high-aspect-ratio BL cells
          (typical inflation layers have AR 50–500:1).  Coupled solves pressure
          and velocity simultaneously in one linear system, which is more robust
          at high AR and at low Re where BL is thick relative to the cell height.
          The cost is ~2× memory per iteration, which is acceptable for 2D.

        Why Least Squares Cell Based (LSCB) gradient:
          Green-Gauss cell-based can produce errors on non-orthogonal meshes
          (typical near LE curvature).  LSCB is more accurate on irregular
          unstructured cells and is the Fluent recommendation for BL-resolving meshes.

        Why Second Order Upwind throughout:
          First-order schemes introduce numerical diffusion that smears the sharp
          Cp peak at the LE suction spike and artificially thickens the BL.
          This makes the LSB appear larger and CD appear higher — not because the
          model is wrong, but because the numerics are smearing the physics.
          Always use Second Order for production runs.
        """
        self.s.setup.general.solver.time = "steady"
        self.s.solution.methods.p_v_coupling.flow_scheme = 'Coupled'
        self.s.solution.methods.spatial_discretization.gradient_scheme = 'least-square-cell-based'

        self.s.solution.methods.spatial_discretization.discretization_scheme['pressure'] = 'second-order'
        self.s.solution.methods.spatial_discretization.discretization_scheme['mom']      = 'second-order-upwind'

        # Turbulent scalar discretization — keys are model-specific
        # 'kt' is NOT a valid spatial_discretization key in Fluent 25.2 — omitted
        _turb_disc = {
            'k-kl-w':         ['kl', 'omega'],
            'transition-sst': ['k', 'omega'],
            'k-omega-sst':    ['k', 'omega'],
        }
        for key in _turb_disc.get(self.turb_model, ['k', 'omega']):
            self.s.solution.methods.spatial_discretization.discretization_scheme[key] = 'second-order-upwind'

        # Coupled explicit p-v URFs
        if conservative:
            self.s.solution.controls.p_v_controls = {
                'explicit_momentum_under_relaxation': 0.5,
                'explicit_pressure_under_relaxation': 0.3,
            }
        else:
            self.s.solution.controls.p_v_controls = {
                'explicit_momentum_under_relaxation': 0.7,
                'explicit_pressure_under_relaxation': 0.5,
            }

        # Pseudo-time relaxation factors for turbulent scalars (via TUI — no direct API)
        std_urfs  = {'density': 1.0, 'k': 0.8, 'kl': 0.8, 'omega': 0.8, 'turb-viscosity': 0.8}
        cons_urfs = {'density': 1.0, 'k': 0.4, 'kl': 0.4, 'omega': 0.5, 'turb-viscosity': 0.6}
        urfs = cons_urfs if conservative else std_urfs
        for var, val in urfs.items():
            self.session.execute_tui(
                f'/solve/set/pseudo-time-method/relaxation-factors/{var} {val}'
            )

        tag = 'conservative' if conservative else 'standard'
        print(f"  Steady solver set ({tag} URFs)")

    # -------------------------------------------------------------------
    # SOLVER METHODS — TRANSIENT
    # -------------------------------------------------------------------

    def _set_methods_transient(self):
        """
        Switch to unsteady-2nd-order, re-enable Coupled (Fluent resets it),
        set transient URFs and time step from LOGIC.md Sec 2.5 / 2.7.
        Returns (dt, chunk_steps, min_chunks, max_chunks).

        Adaptive chunk strategy (replaces fixed washout+avg):
          Instead of a fixed step count, the caller runs chunks of chunk_steps and
          checks CL periodicity after each chunk.  This avoids wasting hours on a
          washout window that is longer than necessary when starting from a converged
          steady solution.  See run_aoa_sweep for the convergence loop.

        Time step selection (Δt = 5×10⁻⁴ × T_conv):
          T_conv = chord / V_inf is the convective time — the time for a fluid parcel
          to traverse one chord length.  The factor 5×10⁻⁴ gives ~2000 steps per
          shedding cycle (typical St 0.1–0.3 → T_shed = T_conv / St).
          This resolves vortex shedding with adequate temporal accuracy.
          If inner iterations exceed 20 per step, halve Δt.

        Chunk size (2 T_conv per chunk):
          Each chunk covers ~2 shedding cycles at St 0.1 — enough to assess
          whether CL has settled into a periodic pattern.  Smaller chunks increase
          overhead from repeated PyFluent calls; larger chunks waste time if the
          solution was already periodic.

        No re-initialization before transient:
          Every transient run is preceded by a converged steady solve, so the field
          is already physically consistent (correct BL, pressure, turbulence quantities).
          Hybrid init would discard that solution and restart from Laplace-based guesses,
          increasing washout time for no benefit.

        Inner iterations (max_iter_per_step = 15):
          Each time step must converge its own sub-iteration loop to ~1e-4 residual
          before advancing.  15 is sufficient for attached flow; at high AoA or during
          active vortex shedding, the inner loop may need more iterations.
          Watch the residual monitor — if it does not reach 1e-4 within 15 iters,
          either reduce Δt or increase max_iter_per_step to 20.

        Fluent resets scheme on time-mode switch:
          When setting solver.time = 'unsteady-2nd-order', Fluent internally resets
          the pressure-velocity coupling back to SIMPLE and clears discretization
          settings.  The re-application of Coupled + LSCB + Second Order after the
          time-mode switch is mandatory, not optional.
        """
        dt          = 5e-4 * self.T_conv
        chunk_steps = int(2.0 * self.T_conv / dt)   # 2 T_conv per check
        min_chunks  = 3                              # run at least 6 T_conv before stopping
        max_chunks  = 20                             # hard cap: 40 T_conv total

        self.s.setup.general.solver.time = "unsteady-2nd-order"

        # Fluent resets coupling and discretization on time-mode switch — re-apply all
        self.s.solution.methods.p_v_coupling.flow_scheme = 'Coupled'
        self.s.solution.methods.spatial_discretization.gradient_scheme = 'least-square-cell-based'
        self.s.solution.methods.spatial_discretization.discretization_scheme['pressure'] = 'second-order'
        self.s.solution.methods.spatial_discretization.discretization_scheme['mom']      = 'second-order-upwind'
        _turb_disc = {
            'k-kl-w':         ['kl', 'omega'],
            'transition-sst': ['k', 'omega'],
            'k-omega-sst':    ['k', 'omega'],
        }
        for key in _turb_disc.get(self.turb_model, ['k', 'omega']):
            self.s.solution.methods.spatial_discretization.discretization_scheme[key] = 'second-order-upwind'

        # Transient URFs (LOGIC.md Sec 2.5 transient set)
        # [? dict keys vary by model — 'kl' only present for k-kl-w]
        self.s.solution.controls.under_relaxation = {
            'k':              0.6,
            'kl':             0.6,
            'omega':          0.7,
            'turb-viscosity': 0.8,
        }

        self.s.solution.run_calculation.transient_controls.time_step_size         = dt
        self.s.solution.run_calculation.transient_controls.max_iter_per_time_step = 15

        print(
            f"  Transient: dt={dt:.4e}s | "
            f"chunk={chunk_steps} steps (2×T_conv) | "
            f"min={min_chunks} chunks | max={max_chunks} chunks"
        )
        return dt, chunk_steps, min_chunks, max_chunks

    # -------------------------------------------------------------------
    # MONITORS
    # -------------------------------------------------------------------

    def _setup_monitors(self, AoA_deg, prefix):
        """
        Create CL and CD report definitions, report files (written every iter/step),
        report plots, and residual criteria.

        Why a new report definition per AoA:
          The force_vector (lift/drag direction) changes with AoA.  Fluent stores
          the vector as part of the report definition object, not as a live expression.
          Creating a new definition per case guarantees the correct projection vector
          for each AoA; reusing and overwriting a single definition works but is
          harder to audit from the output file names.

        Report files vs report plots:
          Report FILES are written to disk every iteration — these are the source
          data for the CL oscillation check and for Strouhal number FFT post-processing.
          Report PLOTS are the GUI monitor windows — they are not saved to disk
          automatically and exist only for visual monitoring during the run.

        Validation compute() call:
          This is a connectivity check only.  The values returned pre-solution are
          meaningless (integrated forces on an uninitialised field).  Its purpose is
          to confirm that the report definition references valid zones and valid fields
          before spending hours on a run that would fail at post-processing.
          If compute() raises, there is a zone name mismatch — check that airfoil_zones
          names match what is in the mesh file.
        """
        p = self._compute_flow_params(AoA_deg)

        cl_name = f"cl_{prefix}"
        cd_name = f"cd_{prefix}"

        # CL definition (lift direction rotates with AoA)
        self.s.solution.report_definitions.lift[cl_name] = {}
        self.s.solution.report_definitions.lift[cl_name] = {
            'force_vector': p['lift_vec'],
            'zones':        self.airfoil_zones,
        }

        # CD definition
        self.s.solution.report_definitions.drag[cd_name] = {}
        self.s.solution.report_definitions.drag[cd_name] = {
            'force_vector': p['drag_vec'],
            'zones':        self.airfoil_zones,
        }

        # Residual criteria
        # k-kl-w equations: continuity, x-velocity, y-velocity, kl, kt, omega
        # transition-sst:   continuity, x-velocity, y-velocity, k, omega, + 2 extra (see below)
        for eq in ['continuity', 'x-velocity', 'y-velocity']:
            self.s.solution.monitor.residual.equations[eq].absolute_criteria = 1e-5
        for eq in ['k', 'kl', 'kt', 'omega']:
            try:
                self.s.solution.monitor.residual.equations[eq].absolute_criteria = 1e-5
            except Exception:
                pass  # equation absent for this model — skip

        # Report files — written every iteration (accumulate history for oscillation check)
        cl_file = str(self.output_dir / f"{cl_name}.out")
        cd_file = str(self.output_dir / f"{cd_name}.out")

        self.s.solution.monitor.report_files[cl_name] = {}
        self.s.solution.monitor.report_files[cl_name] = {
            'report_defs': [cl_name],
            'file_name':   cl_file,
        }
        self.s.solution.monitor.report_files[cd_name] = {}
        self.s.solution.monitor.report_files[cd_name] = {
            'report_defs': [cd_name],
            'file_name':   cd_file,
        }

        # Report plots (GUI monitor windows)
        self.s.solution.monitor.report_plots[cl_name] = {}
        self.s.solution.monitor.report_plots[cl_name] = {
            'report_defs': [cl_name],
            'y_label':     'CL',
            'print':       True,
        }
        self.s.solution.monitor.report_plots[cd_name] = {}
        self.s.solution.monitor.report_plots[cd_name] = {
            'report_defs': [cd_name],
            'y_label':     'CD',
            'print':       True,
        }

        # Transition SST: intermittency (γ) surface average — tracks transition onset per iteration
        # Useful to monitor: once γ stabilises near 1 at a fixed x/c location, transition is steady.
        gamma_name, gamma_file = None, None
        if self.turb_model == 'transition-sst':
            gamma_name = f"gamma_{prefix}"
            self.s.solution.report_definitions.surface[gamma_name] = {}
            self.s.solution.report_definitions.surface[gamma_name] = {
                'report_type':   'surface-areaavg',
                'field':         'intermittency',   # γ field name in Fluent
                'surface_names': self.airfoil_zones,
            }
            gamma_file = str(self.output_dir / f"{gamma_name}.out")
            self.s.solution.monitor.report_files[gamma_name] = {}
            self.s.solution.monitor.report_files[gamma_name] = {
                'report_defs': [gamma_name],
                'file_name':   gamma_file,
            }

        # Validation compute — confirms report defs are correctly set up before running.
        # NOTE: values are meaningless at this point (pre-solution); this is just a connectivity check.
        try:
            defs_to_check = [cl_name, cd_name]
            if gamma_name:
                defs_to_check.append(gamma_name)
            check = self.s.solution.report_definitions.compute(report_defs=defs_to_check)
            print(f"  Report defs validated: {[list(d.keys())[0] for d in check]}")
        except Exception as e:
            print(f"  [WARN] Report def validation failed: {e}")

        return cl_name, cd_name, cl_file, cd_file, gamma_name, gamma_file

    # -------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------

    def _initialize(self):
        """Hybrid initialization, 20 inner iterations."""
        self.s.solution.initialization.initialization_type = 'hybrid'
        self.s.solution.initialization.hybrid_init_options.general_settings = {
            'iter_count': 20
        }
        self.s.solution.initialization.hybrid_initialize()
        if not self._postproc_ready:
            self._setup_postprocessing()
            self._postproc_ready = True
        print("  Hybrid init done.")

    # -------------------------------------------------------------------
    # CL OSCILLATION CHECK (LOGIC.md Sec 2.4)
    # -------------------------------------------------------------------

    def _check_cl_oscillation(self, cl_file):
        """
        Read last 500 entries from CL .out file and assess convergence state.
        Returns: ('converged'|'tighten'|'switch_transient', range, std)

        Why last 500 iterations:
          The first 500–1000 iterations in a new steady case are dominated by
          initialization transient — large swings as the pressure field adjusts.
          Reading only the last 500 avoids misclassifying an otherwise-converging
          case as oscillatory.  At 2000+ total iterations, the last 500 reflects
          the true steady-state character of the solution.

        Threshold rationale (LOGIC.md Sec 2.4):
          range < 0.03 AND std < 0.002  → genuinely flat — steady-state achieved.
          range 0.03–0.08               → marginal oscillation; may be numerical
                                          (URF-driven cycling) or weak physical LSB.
                                          Tighten URFs first to distinguish.
          range > 0.08                  → strong oscillation; physical LSB shedding
                                          cannot be represented by steady RANS.
                                          Switch to transient.
        """
        try:
            rows = []
            with open(cl_file) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith(('(', '#', ')', '"')):
                        continue
                    try:
                        rows.append([float(v) for v in line.split()])
                    except ValueError:
                        continue  # skip any remaining header/label rows
            if len(rows) < 2:
                return 'converged', 0.0, 0.0
            data = np.array(rows)
            cl = data[-500:, 1] if data.shape[0] > 500 else data[:, 1]
        except Exception as e:
            print(f"  [WARN] CL file read error: {e} — assuming converged")
            return 'converged', 0.0, 0.0

        cl_range = float(np.max(cl) - np.min(cl))
        cl_std   = float(np.std(cl))
        print(f"  CL oscillation check: range={cl_range:.4f}, std={cl_std:.5f}")

        if cl_range < 0.03 and cl_std < 0.002:
            return 'converged', cl_range, cl_std
        elif cl_range <= 0.08:
            return 'tighten', cl_range, cl_std
        else:
            return 'switch_transient', cl_range, cl_std

    def _read_history(self, out_file, n_last=None):
        """Load a Fluent .out report file, return (col0, col1) arrays."""
        try:
            rows = []
            with open(out_file) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith(('(', '#', ')', '"')):
                        continue
                    try:
                        rows.append([float(v) for v in line.split()])
                    except ValueError:
                        continue
            if not rows:
                return np.array([0.0]), np.array([0.0])
            data = np.array(rows)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if n_last:
                data = data[-n_last:]
            return data[:, 0], data[:, 1]
        except Exception as e:
            print(f"  [WARN] Could not read {out_file}: {e}")
            return np.array([0.0]), np.array([0.0])

    def _read_transient_history(self, base_file):
        """Collect all Fluent transient data from {stem}_N_1.out files and
        return concatenated (x, y) arrays.

        When dual_time_iterate() is first called Fluent creates _1_1.out and
        immediately begins writing to it — row 0 is the initial condition
        (= the converged steady value), followed by one row per transient
        time step.  _1_1.out IS the transient data for that call; it is NOT
        a read-only copy of the steady file.

        Subsequent dual_time_iterate() calls write to _2_1.out, _3_1.out, …
        (either a new file per call, or Fluent may append — both cases are
        handled by concatenating in index order).

        Rule: include all _N_1.out files (index >= 1) in order.
        Falls back to _read_history(base_file) only if no numbered files
        exist yet (i.e. dual_time_iterate() has not been called at all).
        """
        import re as _re
        base = Path(base_file)
        stem = base.stem
        pattern = _re.compile(rf'^{_re.escape(stem)}_(\d+)_1\.out$', _re.IGNORECASE)
        chunk_files = sorted(
            [p for p in base.parent.glob(f'{stem}_*_1.out') if pattern.match(p.name)],
            key=lambda p: int(pattern.match(p.name).group(1))  # type: ignore[union-attr]
        )

        # All _N_1.out files are transient data (index 1 = first call, 2 = second, …)
        transient_chunks = chunk_files

        if not transient_chunks:
            return self._read_history(base_file)

        x_all, y_all = [], []
        for f in transient_chunks:
            try:
                rows = []
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith(('(', '#', ')', '"')):
                            continue
                        try:
                            parts = [float(v) for v in line.split()]
                            rows.append(parts)
                        except ValueError:
                            continue
                if not rows:
                    continue
                data = np.array(rows)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                x_all.append(data[:, 0])
                y_all.append(data[:, 1])
            except Exception as e:
                print(f"  [WARN] Could not read transient chunk {f.name}: {e}")

        if not y_all:
            return self._read_history(base_file)

        return np.concatenate(x_all), np.concatenate(y_all)

    # -------------------------------------------------------------------
    # POST-PROCESSING SETUP (called once before AoA loop)
    # -------------------------------------------------------------------

    def _setup_postprocessing(self):
        """
        Define contours and streamlines — set up once, re-displayed per AoA case.
        """

        # Cp contour on airfoil surface
        self.s.results.graphics.contour['cp-contour'] = {}
        _c = self.s.results.graphics.contour['cp-contour']
        _c.field = 'pressure-coefficient'
        _c.surfaces_list = self.airfoil_zones
        _c.color_map.title_elements = 'Variable Only'

        # y+ contour
        self.s.results.graphics.contour['yplus-contour'] = {}
        _c = self.s.results.graphics.contour['yplus-contour']
        _c.field = 'y-plus'
        _c.surfaces_list = self.airfoil_zones
        _c.color_map.title_elements = 'Variable Only'

        # Cf (skin friction) contour — LSB shows as negative Cf region
        self.s.results.graphics.contour['cf-contour'] = {}
        _c = self.s.results.graphics.contour['cf-contour']
        _c.field = 'skin-friction-coef'           # confirmed TUI field name
        _c.surfaces_list = self.airfoil_zones
        _c.color_map.title_elements = 'Variable Only'

        # Vorticity magnitude — highlights shear layers and separation bubble
        self.s.results.graphics.contour['vorticity-contour'] = {}
        _c = self.s.results.graphics.contour['vorticity-contour']
        _c.field = 'vorticity-mag'
        _c.surfaces_list = self.airfoil_zones
        _c.color_map.title_elements = 'Variable Only'

        # Stream function — classic 2D topology, identifies closed recirculation (LSB)
        self.s.results.graphics.contour['psi-contour'] = {}
        _c = self.s.results.graphics.contour['psi-contour']
        _c.field = 'stream-function'
        _c.surfaces_list = self.airfoil_zones
        _c.color_map.title_elements = 'Variable Only'

        # Intermittency — Transition SST only.
        # Both 'intermittency' (γ) and 'intermittency-eff' (γ_eff) are confirmed TUI
        # field names, but only appear when the Transition SST model is loaded.
        if self.turb_model == 'transition-sst':
            self.s.results.graphics.contour['intermittency-contour'] = {}
            _c = self.s.results.graphics.contour['intermittency-contour']
            _c.field = 'intermittency'             # γ — transport equation variable
            _c.surfaces_list = self.airfoil_zones
            _c.color_map.title_elements = 'Variable Only'

            self.s.results.graphics.contour['intermittency-eff-contour'] = {}
            _c = self.s.results.graphics.contour['intermittency-eff-contour']
            _c.field = 'intermittency-eff'         # γ_eff — effective (clipped) value used in production terms
            _c.surfaces_list = self.airfoil_zones
            _c.color_map.title_elements = 'Variable Only'

        # Streamlines (pathlines) released backward from airfoil wall.
        # Reverse trace reveals LSB recirculation bubble and separation/reattachment.
        # TUI-confirmed settings: field, surfaces-list, skip, step, options (reverse).
        self.s.results.graphics.pathline['streamlines'] = {}
        _pl = self.s.results.graphics.pathline['streamlines']
        _pl.field                 = 'velocity-magnitude'
        _pl.release_from_surfaces = self.airfoil_zones
        _pl.options.reverse       = True           # backward trace — shows separation
        _pl.step                  = self.chord_m * 0.001   # 0.1%c per step (confirmed key: 'step')
        _pl.skip                  = 5
        _pl.color_map.title_elements = 'Variable Only'

        print("  Post-processing objects set up.")

    def _setup_camera_2d(self):
        """Orthographic camera looking at XY plane, centered on airfoil."""
        v = self.s.results.graphics.views
        v.camera.projection(type='orthographic')
        v.camera.up_vector(xyz=[0., 1., 0.])
        v.camera.position(xyz=[self.chord_m / 2, 0., self.chord_m * 15])
        v.auto_scale()

    def _setup_animation(self, prefix):
        """
        Register solution animations to record during a transient run.
        Called BEFORE dual_time_iterate so Fluent captures frames as it solves.
        Only active when self.animate = True.

        Three animations are registered:
          cp-anim       — pressure coefficient contour  (primary: shows LSB dynamics)
          cf-anim       — skin friction coefficient     (transition / separation front)
          psi-anim      — stream function               (bubble topology per step)

        Frequency = every 5 time steps — enough to show the shedding cycle without
        producing thousands of frames.  At ~2000 steps per T_conv and St~0.2,
        one shedding cycle ≈ 2000/0.2 = 10000 steps → 2000 frames at freq=5.
        Adjust freq upward (e.g. 20) to reduce file size for longer runs.

        Output format: AVI saved per animation name to output_dir.
        Fluent encodes with its built-in writer; no external encoder needed.

        API note: solution_animations key names must match a defined graphics object.
        The display_object_name links to the contour/pathline objects created in
        _setup_postprocessing().  If Fluent raises on display_object_name, try
        using the TUI path /solve/animate instead (see commented alternative below).
        """
        if not self.animate:
            return

        self._setup_camera_2d()   # ensure camera is set before recording

        anim_configs = [
            ('cp-anim',  'cp-contour',  f"anim_cp_{prefix}.avi"),
            ('cf-anim',  'cf-contour',  f"anim_cf_{prefix}.avi"),
            ('psi-anim', 'psi-contour', f"anim_psi_{prefix}.avi"),
        ]

        for anim_name, display_obj, fname in anim_configs:
            self.s.solution.calculation_activity.solution_animations[anim_name] = {}
            _a = self.s.solution.calculation_activity.solution_animations[anim_name]
            _a.frequency        = 5                             # record every 5 time steps
            _a.frequency_of     = 'time-step'                   # 'time-step' or 'iteration'
            _a.display          = display_obj                   # confirmed API name in Fluent 25.2
            _a.storage_type     = 'mpeg'                        # 'mpeg' → AVI in Fluent 2024
            _a.storage_path     = str(self.output_dir / fname)

        print(f"  Animations registered: {[c[0] for c in anim_configs]}")

        # TUI alternative if the settings API above raises:
        # for anim_name, display_obj, fname in anim_configs:
        #     self.session.execute_tui(
        #         f'/solve/animate/add-animation {anim_name} '
        #         f'frequency 5 time-step {display_obj} '
        #         f'mpeg "{str(self.output_dir / fname)}"'
        #     )

    # -------------------------------------------------------------------
    # SAVE PLOTS AND PICTURES PER CASE
    # -------------------------------------------------------------------

    def _save_plots_and_pictures(self, prefix, is_transient):
        """XY plots (Cp, Cf, y+) to .xy files + contour PNG pictures."""
        tag = f"{prefix}_{'T' if is_transient else 'S'}"
        self._setup_camera_2d()

        # XY plots — direction_along_x_axis=[1,0,0] projects onto chord (x/c)
        xy_configs = [
            ('pressure-coefficient', f"cp_{tag}.xy"),
            ('skin-friction-coef',   f"cf_{tag}.xy"),   # confirmed TUI field name
            ('y-plus',               f"yplus_{tag}.xy"),
        ]
        for field, fname in xy_configs:
            self.s.results.plot.plot.plot(
                node_values             = True,
                file_name               = str(self.output_dir / fname),
                order_points            = True,
                y_axis_function         = field,
                y_axis_direction_vector = False,
                y_axis_curve_length     = False,
                x_axis_direction_vector = True,
                direction_along_x_axis  = [1, 0, 0],
                x_axis_curve_length     = False,
                surfaces                = self.airfoil_zones,
            )

        # Contour pictures
        for cont_name, fname in [
            ('cp-contour',       f"cp_contour_{tag}"),
            ('cf-contour',       f"cf_contour_{tag}"),
            ('yplus-contour',    f"yplus_contour_{tag}"),
            ('vorticity-contour',f"vorticity_contour_{tag}"),
            ('psi-contour',      f"psi_contour_{tag}"),
        ]:
            self.s.results.graphics.contour[cont_name].display()
            self.s.results.graphics.picture.save_picture(
                file_name=str(self.output_dir / fname)
            )

        if self.turb_model == 'transition-sst':
            for cont_name, fname in [
                ('intermittency-contour',     f"intermittency_contour_{tag}"),
                ('intermittency-eff-contour', f"intermittency_eff_contour_{tag}"),
            ]:
                self.s.results.graphics.contour[cont_name].display()
                self.s.results.graphics.picture.save_picture(
                    file_name=str(self.output_dir / fname)
                )

        # Streamline picture
        self.s.results.graphics.pathline['streamlines'].display()
        self.s.results.graphics.picture.save_picture(
            file_name=str(self.output_dir / f"streamlines_{tag}")
        )

        print(f"  Plots and pictures saved → {self.output_dir}")

    # -------------------------------------------------------------------
    # SCALAR RESULT EXTRACTION
    # -------------------------------------------------------------------

    def _extract_scalars(self, cl_name, cd_name):
        """Compute CL and CD at the current (steady) solution state."""
        result = self.s.solution.report_definitions.compute(
            report_defs=[cl_name, cd_name]
        )
        # [? compute() returns list of dicts; exact structure may differ by Fluent version]
        cl = result[0][cl_name][0]
        cd = result[0][cd_name][0]
        return float(cl), float(cd)

    # -------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------

    def run_aoa_sweep(self):
        """
        Launch Fluent, set up model once, then loop over AoA_list.
        Per case: steady → oscillation check → transient if needed.
        Returns list of result dicts.
        """
        self._setup_session()
        self._set_turbulence_model()
        # _setup_postprocessing() is called lazily inside _initialize(),
        # after hybrid_initialize() — graphics objects are not active before that.

        self._confirm(
            f"Model: {self.turb_model} | Re={self.Re} | V∞={self.V_inf:.2f} m/s | "
            f"T_conv={self.T_conv:.4f}s\n"
            f"  Check GUI: turbulence model active, reference values correct."
        )

        results = []

        for AoA_deg in self.AoA_list:
            print(f"\n{'='*60}")
            print(f"  AoA = {AoA_deg}°  |  Re = {self.Re}")
            print(f"{'='*60}")

            prefix = (
                f"{self.airfoil_name}_Re{int(self.Re//1000)}k"
                f"_AoA{AoA_deg:.1f}_{self.turb_model}"
            )

            self._set_boundary_conditions(AoA_deg)
            cl_name, cd_name, cl_file, cd_file, gamma_name, gamma_file = self._setup_monitors(AoA_deg, prefix)

            # Geometry pre-check (LOGIC.md Sec 2.3)
            # [TODO] wire t_over_c and cam_over_c from record.json (pipeline stage: geometry)
            # t_over_c   = geom_params['t_over_c']
            # cam_over_c = geom_params['cam_over_c']
            # force_transient = (
            #     self.Re < 100_000 and (t_over_c < 0.10 or cam_over_c > 0.04)
            # )
            force_transient = False

            ran_transient = False

            # ---- STEADY ATTEMPT ----
            if not force_transient:
                self._set_methods_steady()
                self._initialize()

                self._confirm(
                    f"AoA={AoA_deg}° — running 1000 steady iters. "
                    f"Vx={self._compute_flow_params(AoA_deg)['Vx']:.3f} m/s  "
                    f"Vy={self._compute_flow_params(AoA_deg)['Vy']:.3f} m/s"
                )

                print("  Steady: 1000 iters...")
                self.s.solution.run_calculation.iterate(iter_count=1000)

                status, cl_range, _ = self._check_cl_oscillation(cl_file)
                print(f"  → {status}")

                if status == 'converged':
                    print("  Converging — 3000 more iters...")
                    self.s.solution.run_calculation.iterate(iter_count=3000)
                    status, cl_range, _ = self._check_cl_oscillation(cl_file)

                elif status == 'tighten':
                    print("  Oscillating — tightening URFs, 1000 more iters...")
                    self._set_methods_steady(conservative=True)
                    self.s.solution.run_calculation.iterate(iter_count=1000)
                    status, cl_range, _ = self._check_cl_oscillation(cl_file)
                    if status != 'converged':
                        status = 'switch_transient'

                if status == 'switch_transient':
                    print(
                        f"  CL range={cl_range:.3f} exceeds threshold "
                        f"(LOGIC.md Sec 2.4) — switching to transient."
                    )
                    force_transient = True

            # ---- TRANSIENT ----
            if force_transient:
                ran_transient = True
                _, chunk_steps, min_chunks, max_chunks = self._set_methods_transient()

                self._confirm(
                    f"AoA={AoA_deg}° — transient: adaptive chunks of {chunk_steps} steps "
                    f"(2×T_conv each), min={min_chunks}, max={max_chunks}.\n"
                    f"  Check GUI: unsteady solver active, Coupled scheme, dt correct."
                )

                self._set_boundary_conditions(AoA_deg)  # re-confirm after mode switch
                self._setup_animation(prefix)           # no-op if animate=False

                # ----------------------------------------------------------
                # Adaptive chunk loop: run 2 T_conv at a time, check CL
                # periodicity after each chunk.  Stop as soon as two
                # consecutive chunks show stable std (periodic shedding).
                # This avoids committing to a fixed washout duration — since
                # we start from a converged steady field the actual washout
                # is much shorter than the old conservative estimate.
                # ----------------------------------------------------------
                prev_std   = None
                total_ran  = 0
                converged  = False

                for chunk_idx in range(max_chunks):
                    print(f"  Chunk {chunk_idx + 1}/{max_chunks} ({chunk_steps} steps)...")
                    self.s.solution.run_calculation.dual_time_iterate(
                        time_step_count   = chunk_steps,
                        max_iter_per_step = 15,
                    )
                    total_ran += chunk_steps

                    _, cl_hist = self._read_transient_history(cl_file)
                    if len(cl_hist) < 10:
                        continue  # not enough data yet

                    # Assess last chunk's std and compare to previous chunk
                    window    = cl_hist[-chunk_steps:] if len(cl_hist) >= chunk_steps else cl_hist
                    curr_std  = float(np.std(window))
                    cl_range  = float(np.max(window) - np.min(window))
                    prev_std_str = f'{prev_std:.5f}' if prev_std is not None else 'n/a'
                    print(f"    CL std={curr_std:.5f}  range={cl_range:.4f}  "
                          f"(prev_std={prev_std_str})")

                    if chunk_idx + 1 >= min_chunks:
                        if prev_std is not None and abs(curr_std - prev_std) / (prev_std + 1e-9) < 0.05:
                            print(f"  Periodic: std stable over 2 chunks — stopping at chunk {chunk_idx + 1}.")
                            converged = True
                            break

                    prev_std = curr_std

                if not converged:
                    print(f"  [WARN] Max chunks ({max_chunks}) reached without clear periodicity.")

                # Average over the last chunk only (already in periodic regime)
                _, cl_hist = self._read_transient_history(cl_file)
                _, cd_hist = self._read_transient_history(cd_file)
                cl_hist = cl_hist[-chunk_steps:] if len(cl_hist) >= chunk_steps else cl_hist
                cd_hist = cd_hist[-chunk_steps:] if len(cd_hist) >= chunk_steps else cd_hist

                cl_mean, cl_std = float(np.mean(cl_hist)), float(np.std(cl_hist))
                cd_mean, cd_std = float(np.mean(cd_hist)), float(np.std(cd_hist))
                print(f"  CL = {cl_mean:.4f} ± {cl_std:.4f}")
                print(f"  CD = {cd_mean:.5f} ± {cd_std:.5f}")
                print(f"  Total steps run: {total_ran}")

                # Intermittency (Transition SST only)
                gamma_mean = None
                if gamma_name and gamma_file:
                    _, gamma_hist = self._read_transient_history(gamma_file)
                    gamma_hist = gamma_hist[-chunk_steps:] if len(gamma_hist) >= chunk_steps else gamma_hist
                    gamma_mean = float(np.mean(gamma_hist))
                    print(f"  γ (intermittency, surface avg) = {gamma_mean:.4f}")

                result = {
                    'AoA': AoA_deg, 'Re': self.Re, 'model': self.turb_model,
                    'CL': cl_mean,  'CD': cd_mean,
                    'CL_std': cl_std, 'CD_std': cd_std,
                    'gamma_mean': gamma_mean,
                    'transient': True,
                    'chunks_run': total_ran // chunk_steps,
                    'total_steps': total_ran,
                }

            else:
                cl_val, cd_val = self._extract_scalars(cl_name, cd_name)
                print(f"  CL = {cl_val:.4f}")
                print(f"  CD = {cd_val:.5f}")

                # Intermittency (Transition SST only)
                gamma_val = None
                if gamma_name:
                    try:
                        check = self.s.solution.report_definitions.compute(
                            report_defs=[gamma_name]
                        )
                        gamma_val = float(check[0][gamma_name][0])
                        print(f"  γ (intermittency, surface avg) = {gamma_val:.4f}")
                    except Exception as e:
                        print(f"  [WARN] gamma compute failed: {e}")

                result = {
                    'AoA': AoA_deg, 'Re': self.Re, 'model': self.turb_model,
                    'CL': cl_val, 'CD': cd_val,
                    'CL_std': 0.0, 'CD_std': 0.0,
                    'gamma_mean': gamma_val,
                    'transient': False,
                }

            # ---- POST-PROCESSING ----
            self._confirm(
                f"AoA={AoA_deg}° solved. About to save XY plots and contour pictures."
            )
            try:
                self._save_plots_and_pictures(prefix, ran_transient)
            except Exception as e:
                print(f"  [WARN] Post-processing failed for AoA={AoA_deg}°: {e}")
                print(f"  [WARN] Skipping plots/pictures — continuing to next AoA.")

            # Save case + data snapshot per AoA
            case_file = str(self.output_dir / f"case_{prefix}.cas.h5")
            try:
                self.s.file.write(file_type='case-data', file_name=case_file)
                print(f"  Case saved: {case_file}")
            except Exception as e:
                print(f"  [WARN] Case save failed for AoA={AoA_deg}°: {e}")

            results.append(result)

        # ---- END OF SWEEP ----
        self._confirm("All AoA cases complete. Ready to exit Fluent.")
        self.session.exit(wait=True)
        elapsed = time.time() - self._t0
        h, rem = divmod(int(elapsed), 3600)
        m, s   = divmod(rem, 60)

        print(f"\n{'='*60}")
        print(f"  Polar — {self.airfoil_name}  Re={self.Re}  {self.turb_model}")
        print(f"  Total time (launch → exit): {h:02d}h {m:02d}m {s:02d}s")
        print(f"{'='*60}")
        for r in results:
            tag = 'T' if r['transient'] else 'S'
            line = (
                f"  [{tag}] AoA={r['AoA']:5.1f}°  "
                f"CL={r['CL']:.4f}  CD={r['CD']:.5f}"
            )
            if r['transient']:
                line += f"  ±CL={r['CL_std']:.4f}"
            print(line)

        return results


# =======================================================================
# PARAMETRIC STUDY — experimental, not tested
# =======================================================================
#
# Fluent's parametric study feature lets you define design points and run
# them in one session. The snag for airfoils: force direction vectors also
# change with AoA, so you cannot just parameterize Vx/Vy alone — the CL/CD
# definitions need to update too. Possible approach:
#
#   1. Define Vx, Vy, lift_x, lift_y, drag_x, drag_y as named expressions
#      with input_parameter = True
#   2. Reference them in BCs and force report definitions
#   3. Create design points for each (AoA, Re) pair
#   4. Call session.settings.parametric_study.run_all()
#
# Sketch (needs testing):
#
#   session.settings.setup.named_expressions['Vx'] = {
#       'definition': '9.0[m s^-1]', 'input_parameter': True,
#   }
#   session.settings.setup.named_expressions['Vy'] = {
#       'definition': '0.0[m s^-1]', 'input_parameter': True,
#   }
#   # Reference in BC — [? strings as expression names in velocity_components]
#   session.settings.setup.boundary_conditions.velocity_inlet['inlet'] = {
#       'momentum': {
#           'velocity_specification_method': 'Components',
#           'velocity_components': ['Vx', 'Vy'],
#       }
#   }
#   # [? session.settings.parametric_study API not confirmed — may need pyfluent docs]
#
# =======================================================================


# =======================================================================
# USAGE EXAMPLES
# =======================================================================
#
# --- Re 61k, k-kL-ω ---
#
# from core.solver.SOLVER import FluentAirfoilSolver
#
# solver = FluentAirfoilSolver(
#     mesh_path      = r"data/runs/NACA6409/meshes/Fine_NACA6409_2d.msh.h5",
#     airfoil_name   = "NACA6409",
#     chord_m        = 0.100,
#     Re             = 61_000,
#     AoA_list       = [0, 2, 4, 6, 8, 10, 12, 14],
#     turb_model     = "k-kl-w",
#     output_dir     = r"data/runs/NACA6409/results/Re61k_kkl",
#     processor_count= 8,
# )
# results = solver.run_aoa_sweep()
#
# --- Re 200k, Transition SST ---
#
# solver = FluentAirfoilSolver(
#     mesh_path    = r"data/runs/NACA6409/meshes/Fine_NACA6409_2d.msh.h5",
#     airfoil_name = "NACA6409",
#     chord_m      = 0.100,
#     Re           = 200_600,
#     AoA_list     = [0, 2, 4, 6, 8, 10],
#     turb_model   = "transition-sst",
#     output_dir   = r"data/runs/NACA6409/results/Re200k_tsst",
# )
# results = solver.run_aoa_sweep()
#
# --- Strouhal from a transient CL file (after transient run) ---
#
# import numpy as np
# from scipy.fft import fft, fftfreq
# from pathlib import Path
#
# cl_file  = Path(r"data/runs/NACA6409/results/Re61k_kkl") / \
#            "cl_NACA6409_Re61k_AoA8.0_k-kl-w.out"
# data     = np.loadtxt(cl_file, comments=['(', '#', ')'])
# cl       = data[:, 1]
# cl       = cl[int(len(cl) * 0.3):]          # trim initial transient
# dt       = 5e-4 * (0.100 / (61_000 * 1.48e-5 / 0.100))   # dt from T_conv
# freqs    = fftfreq(len(cl), dt)
# fft_v    = np.abs(fft(cl))
# pos      = freqs > 0
# f_peak   = freqs[pos][np.argmax(fft_v[pos])]
# V_inf    = 61_000 * 1.48e-5 / 0.100         # = 9.03 m/s
# St       = f_peak * 0.100 / V_inf
# print(f"St = {St:.4f}  (expected 0.1–0.3 for LSB shedding)")
#
# =======================================================================

# Validation:
# solver = FluentAirfoilSolver(
#     mesh_path    = r"data/runs/NACA6409/meshes/Fine_NACA6409_2d.msh.h5",
#     airfoil_name = "NACA6409",
#     chord_m      = 1.000,
#     Re           = 61_000,
#     AoA_list     = [0.26, 5.00],
#     turb_model   = "k-kl-w",
#     output_dir   = r"data/runs/NACA6409/results/Re61k_kkl",
#     animate      = True,   # animation fires only if the case goes transient
#                            # → no effect on 0.26° if it converges steady
# )

if __name__ == "__main__":
    solver = FluentAirfoilSolver(
        mesh_path    = r"data/runs/NACA6409/meshes/coarse_NACA6409.msh.h5",
        airfoil_name = "NACA6409",
        chord_m      = 1.000,
        Re           = 61_000,
        AoA_list     = [0.26, 5.00],
        turb_model   = "k-kl-w",
        output_dir   = r"data/runs/NACA6409/results/Re61k",
        animate      = False,
        debug        = True,   # True  → pauses at each checkpoint for inspection
                                # False → runs fully automated, no prompts
    )
    results = solver.run_aoa_sweep()
    print(results)