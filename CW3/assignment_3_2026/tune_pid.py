"""
Bayesian optimisation of position and yaw PID gains.

Run from the assignment_3_2026 directory:
    python tune_pid.py

Requires: scikit-optimize  (pip install scikit-optimize)
"""

import argparse
import json
import math
import numpy as np
import pybullet as p
import pybullet_data

from src.PID_controller import PIDController
from src.tello_controller import TelloController
from src.wind import Wind


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SIM_HZ          = 1000          # physics steps per second
CTRL_HZ         = 50            # position controller Hz
SIM_SECONDS     = 10.0          # match professor's 10-second test window
STEADY_SECONDS  = 3.0           # score only the last N seconds (steady state)
TARGET          = (2.0, 2.0, 2.0, 0.0)    # furthest position target, no yaw — tune position first

DRONE_URDF      = "resources/tello.urdf"
START_POS       = [0, 0, 1]

M   = 0.088
L   = 0.06
KF  = 0.566e-5
KM  = 0.762e-7
K_TRANS = np.array([3.365e-2, 3.365e-2, 3.365e-2])
TM  = 0.0163


# ---------------------------------------------------------------------------
# One headless simulation trial
# ---------------------------------------------------------------------------
def run_trial(kp, ki, kd, kp_yaw, ki_yaw, kd_yaw, ki_sat=1.0, verbose=False, gui=False, target_override=None):
    """
    Run a headless PyBullet simulation with the given position and yaw PID gains.
    Returns the mean combined error (position + yaw) over the last STEADY_SECONDS.
    """
    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=physics_client)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.loadURDF("plane.urdf", physicsClientId=physics_client)

    start_orn = p.getQuaternionFromEuler([0, 0, 0])
    drone_id = p.loadURDF(DRONE_URDF, START_POS, start_orn,
                          physicsClientId=physics_client)

    # Build PID controllers with candidate gains
    kp_arr  = np.array([kp,     kp,     kp    ])
    ki_arr  = np.array([ki,     ki,     ki    ])
    kd_arr  = np.array([kd,     kd,     kd    ])
    sat_arr = np.array([ki_sat, ki_sat, ki_sat])

    pos_pid = PIDController(Kp=kp_arr, Ki=ki_arr, Kd=kd_arr, Ki_sat=sat_arr)

    yaw_pid = PIDController(
        Kp=np.array([kp_yaw, 0.0, 0.0]),
        Ki=np.array([ki_yaw, 0.0, 0.0]),
        Kd=np.array([kd_yaw, 0.0, 0.0]),
        Ki_sat=np.array([ki_sat, 0.0, 0.0]),
    )

    tello = TelloController(9.81, M, L, 0.35, KF, KM)

    timestep      = 1.0 / SIM_HZ
    ctrl_timestep = 1.0 / CTRL_HZ
    steps_per_ctrl = int(round(ctrl_timestep / timestep))
    total_steps    = int(SIM_SECONDS * SIM_HZ)
    steady_start   = total_steps - int(STEADY_SECONDS * SIM_HZ)

    VEL_LIMIT = 3.5
    YAW_LIMIT = 1.5

    prev_rpm     = np.zeros(4)
    desired_vel  = np.zeros(3)
    yaw_rate_sp  = 0.0
    loop_counter = 0
    pos_errors = []
    yaw_errors = []

    tgt_x, tgt_y, tgt_z, tgt_yaw = target_override if target_override is not None else TARGET

    for step in range(total_steps):
        pos, quat = p.getBasePositionAndOrientation(drone_id,
                                                    physicsClientId=physics_client)
        lin_vel_world, ang_vel_world = p.getBaseVelocity(drone_id,
                                                         physicsClientId=physics_client)

        roll, pitch, yaw = p.getEulerFromQuaternion(quat)
        yaw_quat = p.getQuaternionFromEuler([0, 0, yaw])
        _, inv_quat_yaw = p.invertTransform([0, 0, 0], yaw_quat)
        _, inv_quat     = p.invertTransform([0, 0, 0], quat)

        lin_vel = np.array(p.rotateVector(inv_quat_yaw, lin_vel_world))
        ang_vel = np.array(p.rotateVector(inv_quat,     ang_vel_world))

        loop_counter += 1
        if loop_counter >= steps_per_ctrl:
            loop_counter = 0

            # Position error
            pos_error = np.array([tgt_x - pos[0],
                                  tgt_y - pos[1],
                                  tgt_z - pos[2]])

            # Position PID → velocity command (world frame)
            vel_world = pos_pid.control_update(pos_error, ctrl_timestep)
            vel_world = np.clip(vel_world, -VEL_LIMIT, VEL_LIMIT)

            # Yaw PID
            yaw_error = tgt_yaw - yaw
            yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
            yaw_rate_sp = float(np.clip(
                yaw_pid.control_update(np.array([yaw_error, 0, 0]), ctrl_timestep)[0],
                -YAW_LIMIT, YAW_LIMIT
            ))

            desired_vel = vel_world

            # Record error in steady-state window
            if step >= steady_start:
                pos_errors.append(np.linalg.norm(pos_error))
                yaw_errors.append(abs(yaw_error))

        # Motor dynamics + physics
        rpm = tello.compute_control(desired_vel, lin_vel, quat, ang_vel,
                                    yaw_rate_sp, timestep)

        rpm_deriv = (rpm - prev_rpm) / TM
        rpm = prev_rpm + rpm_deriv * timestep
        prev_rpm = rpm

        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        omega = rpm * (2 * np.pi / 60)
        motor_forces = omega**2 * KF
        thrust = np.array([0, 0, np.sum(motor_forces)])
        vel_body = rotation.T @ lin_vel_world
        drag_body = -K_TRANS * vel_body
        force = drag_body + thrust

        z_torques = omega**2 * KM
        z_torque  = -z_torques[0] - z_torques[1] + z_torques[2] + z_torques[3]
        x_torque  = (-motor_forces[0] + motor_forces[1] + motor_forces[2] - motor_forces[3]) * L
        y_torque  = (-motor_forces[0] + motor_forces[1] - motor_forces[2] + motor_forces[3]) * L
        torques = np.array([x_torque, y_torque, z_torque])

        p.applyExternalForce(drone_id, -1, force.tolist(), [0, 0, 0],
                             p.LINK_FRAME, physicsClientId=physics_client)
        p.applyExternalTorque(drone_id, -1, torques.tolist(),
                              p.LINK_FRAME, physicsClientId=physics_client)
        p.stepSimulation(physicsClientId=physics_client)

    p.disconnect(physics_client)

    mean_pos_err = float(np.mean(pos_errors)) if pos_errors else 9999.0
    mean_yaw_err = float(np.mean(yaw_errors)) if yaw_errors else 9999.0
    # Combine: yaw weighted by 0.5 (yaw error is in radians, pos in metres)
    score = mean_pos_err + 0.5 * mean_yaw_err
    if verbose:
        print(f"  pos={mean_pos_err:.4f} m  yaw={mean_yaw_err:.4f} rad  score={score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Bayesian optimisation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Show convergence plot after optimisation")
    parser.add_argument("--n-calls", type=int, default=100, help="Total BO evaluations (default 100)")
    parser.add_argument("--yaw-only", action="store_true",
                        help="Fix pos gains from best_gains.json and only tune yaw gains. "
                             "Uses a 90-deg yaw target so yaw error is meaningful.")
    args = parser.parse_args()

    try:
        from skopt import gp_minimize
        from skopt.space import Real
        from skopt.utils import use_named_args
    except ImportError:
        print("ERROR: scikit-optimize not installed. Run:  pip install scikit-optimize")
        return

    if args.yaw_only:
        # Load fixed position gains
        try:
            with open("best_gains.json") as f:
                saved = json.load(f)
        except FileNotFoundError:
            print("ERROR: best_gains.json not found. Run without --yaw-only first.")
            return
        fixed_kp  = saved["kp"]
        fixed_ki  = saved["ki"]
        fixed_kd  = saved["kd"]

        # Use a 90-degree yaw target so the yaw controller must actually work
        yaw_target = (TARGET[0], TARGET[1], TARGET[2], 0.5)  # ~28 deg — large enough to matter, small enough not to destabilise position

        space = [
            Real(0.1, 5.0,  name="kp_yaw"),
            Real(0.0, 0.05, name="ki_yaw"),
            Real(0.0, 0.5,  name="kd_yaw"),
        ]

        call_count = [0]

        @use_named_args(space)
        def objective(kp_yaw, ki_yaw, kd_yaw):
            call_count[0] += 1
            print(f"Trial {call_count[0]:3d}  "
                  f"kp_yaw={kp_yaw:.3f} ki_yaw={ki_yaw:.4f} kd_yaw={kd_yaw:.3f}", end="  ")
            score = run_trial(fixed_kp, fixed_ki, fixed_kd, kp_yaw, ki_yaw, kd_yaw,
                              verbose=False, target_override=yaw_target)
            print(f"→ {score:.4f}")
            return score

        print("=" * 55)
        print("Yaw-only BO — pos gains fixed, target:", yaw_target)
        print(f"  Fixed pos: kp={fixed_kp:.4f} ki={fixed_ki:.4f} kd={fixed_kd:.4f}")
        print("=" * 55)

        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=args.n_calls,
            n_initial_points=10,
            acq_func="EI",
            random_state=42,
            verbose=False,
        )

        best_kp_yaw, best_ki_yaw, best_kd_yaw = result.x
        best_kp, best_ki, best_kd = fixed_kp, fixed_ki, fixed_kd

    else:
        yaw_target = TARGET

        # Search space: position gains + yaw gains
        # NOTE: kp/kd upper bounds widened — previous run hit the boundary at 2.0/0.5
        # ki bounds kept narrow — trials show ki≈0 consistently wins
        space = [
            Real(0.1, 5.0,  name="kp"),
            Real(0.0, 0.05, name="ki"),
            Real(0.0, 1.5,  name="kd"),
            Real(0.1, 5.0,  name="kp_yaw"),
            Real(0.0, 0.05, name="ki_yaw"),
            Real(0.0, 0.5,  name="kd_yaw"),
        ]

        call_count = [0]

        @use_named_args(space)
        def objective(kp, ki, kd, kp_yaw, ki_yaw, kd_yaw):
            call_count[0] += 1
            print(f"Trial {call_count[0]:3d}  "
                  f"kp={kp:.3f} ki={ki:.4f} kd={kd:.3f}  "
                  f"kp_yaw={kp_yaw:.3f} ki_yaw={ki_yaw:.4f} kd_yaw={kd_yaw:.3f}", end="  ")
            score = run_trial(kp, ki, kd, kp_yaw, ki_yaw, kd_yaw, verbose=False)
            print(f"→ {score:.4f}")
            return score

        print("=" * 55)
        print("Bayesian PID optimisation — target:", TARGET)
        print("=" * 55)

        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=args.n_calls,
            n_initial_points=20,
            acq_func="EI",
            random_state=42,
            verbose=False,
        )

        best_kp, best_ki, best_kd, best_kp_yaw, best_ki_yaw, best_kd_yaw = result.x

    print("\n" + "=" * 55)
    print(f"BEST gains found (score={result.fun:.4f}):")
    print(f"  Pos  Kp={best_kp:.4f}  Ki={best_ki:.4f}  Kd={best_kd:.4f}")
    print(f"  Yaw  Kp={best_kp_yaw:.4f}  Ki={best_ki_yaw:.4f}  Kd={best_kd_yaw:.4f}")
    print("=" * 55)
    print("\nPaste these into controller.py:")
    print(f"Kp_pos   = np.array([{best_kp:.4f}, {best_kp:.4f}, {best_kp:.4f}])")
    print(f"Ki_pos   = np.array([{best_ki:.4f}, {best_ki:.4f}, {best_ki:.4f}])")
    print(f"Kd_pos   = np.array([{best_kd:.4f}, {best_kd:.4f}, {best_kd:.4f}])")
    print(f"Kp_yaw   = np.array([{best_kp_yaw:.4f}, 0.0, 0.0])")
    print(f"Ki_yaw   = np.array([{best_ki_yaw:.4f}, 0.0, 0.0])")
    print(f"Kd_yaw   = np.array([{best_kd_yaw:.4f}, 0.0, 0.0])")

    # Save best gains to JSON for demo.py to load
    gains = {
        "kp": best_kp, "ki": best_ki, "kd": best_kd,
        "kp_yaw": best_kp_yaw, "ki_yaw": best_ki_yaw, "kd_yaw": best_kd_yaw,
        "score": result.fun,
    }
    with open("best_gains.json", "w") as f:
        json.dump(gains, f, indent=2)
    print("\nSaved to best_gains.json")

    # Verify
    print("\nVerifying best gains...")
    run_trial(best_kp, best_ki, best_kd, best_kp_yaw, best_ki_yaw, best_kd_yaw,
              verbose=True, target_override=yaw_target)

    if args.plot:
        import matplotlib.pyplot as plt
        from skopt.plots import plot_convergence, plot_objective

        # Convergence curve
        plot_convergence(result)
        plt.tight_layout()
        plt.savefig("bo_convergence.png", dpi=150)
        print("Convergence plot saved to bo_convergence.png")

        # Response surfaces — GP's learned objective landscape
        # Shows how score changes across each pair of gains (others fixed at optimum)
        fig = plot_objective(result, n_points=20)
        plt.suptitle("Response surfaces — GP surrogate objective", y=1.01)
        plt.savefig("bo_response_surfaces.png", dpi=150, bbox_inches="tight")
        print("Response surfaces saved to bo_response_surfaces.png")

        plt.show()


if __name__ == "__main__":
    main()

# EXAMPLE USE:
#   python tune_pid.py --n-calls 50
#   python tune_pid.py --n-calls 200 --plot

# OUTPUT 
#   best_gains.json
#   bo_convergence.png