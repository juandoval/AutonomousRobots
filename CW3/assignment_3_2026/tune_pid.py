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
SIM_SECONDS     = 20.0          # 10s approach + 10s scoring, matches professor's test
STEADY_SECONDS  = 10.0          # score the last 10s (the measurement window)

# Representative targets — covers far reach, 3-D diagonal, and yaw demand
# Averaged to get gains that generalise across the 50-trial marking sweep
EVAL_TARGETS = [
    (4.0,  0.0,  1.0, 0.0 ),   # far lateral, no yaw
    (2.0,  2.0,  2.0, 0.0 ),   # 3-D diagonal
    (1.0, -1.0,  0.5, 1.57),   # short target with large yaw demand
]

TARGET = EVAL_TARGETS[0]   # kept for backward-compat with --yaw-only branch

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
def run_trial(kp, ki, kd, kp_yaw, ki_yaw, kd_yaw, ki_sat=1.0, dob_bw=3.0,
              verbose=False, gui=False, target_override=None, wind_enabled=True,
              wind_seed=42):
    """
    Run a headless PyBullet simulation with the given position/yaw PID gains and
    DOB bandwidth.  Returns the mean combined error over the last STEADY_SECONDS.
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

    import random as _random
    _random.seed(wind_seed)
    wind_sim = Wind(max_steady_state=0.02, max_gust=0.02, k_gusts=0.1) if wind_enabled else None

    timestep      = 1.0 / SIM_HZ
    ctrl_timestep = 1.0 / CTRL_HZ
    steps_per_ctrl = int(round(ctrl_timestep / timestep))
    total_steps    = int(SIM_SECONDS * SIM_HZ)
    steady_start   = total_steps - int(STEADY_SECONDS * SIM_HZ)

    VEL_LIMIT = 3.5
    YAW_LIMIT = 1.5

    # DOB state (mirrors controller.py)
    dob_prev_pos  = None
    dob_d_hat     = np.zeros(3)
    dob_prev_vcmd = np.zeros(3)
    dob_alpha     = (2 * math.pi * dob_bw * ctrl_timestep) / (1 + 2 * math.pi * dob_bw * ctrl_timestep)

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

            cur_pos = np.array(pos)

            # Position error
            pos_error = np.array([tgt_x - cur_pos[0],
                                  tgt_y - cur_pos[1],
                                  tgt_z - cur_pos[2]])

            # Nominal PD → velocity command (world frame)
            vel_nom = pos_pid.control_update(pos_error, ctrl_timestep)

            # DOB: estimate and cancel disturbance
            if dob_prev_pos is not None:
                v_est = (cur_pos - dob_prev_pos) / ctrl_timestep
                d_raw = v_est - dob_prev_vcmd
                if np.linalg.norm(d_raw) > 2.0:
                    dob_d_hat[:] = 0.0
                else:
                    dob_d_hat[:] = dob_alpha * d_raw + (1 - dob_alpha) * dob_d_hat
            dob_prev_pos = cur_pos.copy()

            vel_world = np.clip(vel_nom - dob_d_hat, -VEL_LIMIT, VEL_LIMIT)
            dob_prev_vcmd = vel_world.copy()

            # Yaw PID
            yaw_error = tgt_yaw - yaw
            yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
            yaw_rate_sp = float(np.clip(
                yaw_pid.control_update(np.array([yaw_error, 0, 0]), ctrl_timestep)[0],
                -YAW_LIMIT, YAW_LIMIT
            ))

            # World → body frame (same rotation as controller.py)
            vx_b = vel_world[0] * math.cos(yaw) + vel_world[1] * math.sin(yaw)
            vy_b = -vel_world[0] * math.sin(yaw) + vel_world[1] * math.cos(yaw)
            desired_vel = np.array([vx_b, vy_b, vel_world[2]])

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
        if wind_sim is not None:
            wind_force = wind_sim.get_wind(timestep)
            p.applyExternalForce(drone_id, -1, wind_force.tolist(), list(pos),
                                 p.WORLD_FRAME, physicsClientId=physics_client)
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
# Objective: average score across all eval targets (with wind)
# ---------------------------------------------------------------------------
_trial_count = [0]

def objective(params):
    kp, ki, kd, kp_yaw, ki_yaw, kd_yaw, dob_bw = params

    # Hard bounds — reject out-of-range params (Nelder-Mead can wander)
    # Kp lower bound = 0.6 ensures settling to 0.01m in <10s for any 4m target
    LB = np.array([0.6, 0.0,  0.0, 0.5, 0.0,  0.0,  0.5])
    UB = np.array([1.0, 0.05, 0.8, 5.0, 0.05, 0.5, 10.0])
    if np.any(params < LB) or np.any(params > UB):
        return 9999.0

    _trial_count[0] += 1
    print(f"Trial {_trial_count[0]:3d}  "
          f"kp={kp:.3f} ki={ki:.4f} kd={kd:.3f}  "
          f"kp_yaw={kp_yaw:.3f} ki_yaw={ki_yaw:.4f} kd_yaw={kd_yaw:.3f}  "
          f"dob_bw={dob_bw:.2f}", end="  ")

    scores = [
        run_trial(kp, ki, kd, kp_yaw, ki_yaw, kd_yaw, dob_bw=dob_bw,
                  wind_enabled=True, wind_seed=42 + i,
                  target_override=tgt, verbose=False)
        for i, tgt in enumerate(EVAL_TARGETS)
    ]
    score = float(np.mean(scores))
    print(f"→ {score:.4f}  {[f'{s:.4f}' for s in scores]}")
    return score


def print_results(best_params, best_score):
    kp, ki, kd, kp_yaw, ki_yaw, kd_yaw, dob_bw = best_params
    print("\n" + "=" * 55)
    print(f"BEST gains found (score={best_score:.4f}):")
    print(f"  Pos  Kp={kp:.4f}  Ki={ki:.4f}  Kd={kd:.4f}")
    print(f"  Yaw  Kp={kp_yaw:.4f}  Ki={ki_yaw:.4f}  Kd={kd_yaw:.4f}")
    print(f"  DOB  bandwidth={dob_bw:.4f} Hz")
    print("=" * 55)
    print("\nPaste these into controller.py:")
    print(f"Kp_pos        = np.array([{kp:.4f}, {kp:.4f}, {kp:.4f}])")
    print(f"Ki_pos        = np.array([{ki:.4f}, {ki:.4f}, {ki:.4f}])")
    print(f"Kd_pos        = np.array([{kd:.4f}, {kd:.4f}, {kd:.4f}])")
    print(f"Kp_yaw        = np.array([{kp_yaw:.4f}, 0.0, 0.0])")
    print(f"Ki_yaw        = np.array([{ki_yaw:.4f}, 0.0, 0.0])")
    print(f"Kd_yaw        = np.array([{kd_yaw:.4f}, 0.0, 0.0])")
    print(f"DOB_BANDWIDTH = {dob_bw:.4f}")
    gains = {
        "kp": kp, "ki": ki, "kd": kd,
        "kp_yaw": kp_yaw, "ki_yaw": ki_yaw, "kd_yaw": kd_yaw,
        "dob_bw": dob_bw, "score": best_score,
    }
    with open("best_gains.json", "w") as f:
        json.dump(gains, f, indent=2)
    print("\nSaved to best_gains.json")
    print("\nVerifying across all targets:")
    for i, tgt in enumerate(EVAL_TARGETS):
        print(f"  Target {i+1} {tgt}:", end="")
        run_trial(kp, ki, kd, kp_yaw, ki_yaw, kd_yaw, dob_bw=dob_bw,
                  wind_enabled=True, wind_seed=42 + i,
                  verbose=True, target_override=tgt)


# ---------------------------------------------------------------------------
# Optimisation — Nelder-Mead (scipy) with random-search fallback
# ---------------------------------------------------------------------------
def tune_nelder_mead(x0, max_iter):
    """Nelder-Mead simplex: fast, no extra libraries beyond scipy."""
    from scipy.optimize import minimize
    print("=" * 55)
    print("Nelder-Mead simplex optimisation (scipy)")
    print("=" * 55)
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-4,
                            "disp": True, "adaptive": True})
    return res.x, res.fun


def tune_random_search(n_trials, rng):
    """Pure-numpy random search — no extra libraries needed."""
    print("=" * 55)
    print(f"Random search — {n_trials} trials (numpy only)")
    print("=" * 55)
    # Search bounds:  kp   ki     kd   kp_y  ki_y   kd_y  dob_bw
    LB = np.array([0.6, 0.0,  0.0,  0.5,  0.0,  0.0,  0.5])
    UB = np.array([1.0, 0.05, 0.8,  5.0,  0.05, 0.5, 10.0])
    best_score = 9999.0
    best_params = (LB + UB) / 2
    for _ in range(n_trials):
        params = rng.uniform(LB, UB)
        score = objective(params)
        if score < best_score:
            best_score = score
            best_params = params.copy()
    return best_params, best_score


def main():
    parser = argparse.ArgumentParser(
        description="Tune DOB+PID gains for the Tello controller. "
                    "Uses Nelder-Mead (scipy) if available, otherwise random search.")
    parser.add_argument("--n-iter", type=int, default=200,
                        help="Max iterations / trials (default 200)")
    parser.add_argument("--random-only", action="store_true",
                        help="Force random search even if scipy is available")
    args = parser.parse_args()

    # Starting point: analytically safe values (Kp<VEL_LIMIT/4m, no saturation)
    x0 = np.array([0.80, 0.00, 0.25, 2.50, 0.01, 0.05, 2.0])
    #               kp    ki    kd   kp_y  ki_y  kd_y  dob_bw

    rng = np.random.default_rng(42)

    if args.random_only:
        best_params, best_score = tune_random_search(args.n_iter, rng)
    else:
        try:
            from scipy.optimize import minimize as _scipy_minimize
            del _scipy_minimize
            best_params, best_score = tune_nelder_mead(x0, args.n_iter)
        except ImportError:
            print("scipy not found — falling back to random search")
            best_params, best_score = tune_random_search(args.n_iter, rng)

    print_results(best_params, best_score)


if __name__ == "__main__":
    main()

# USAGE:
#   python tune_pid.py                    # Nelder-Mead if scipy available, else random
#   python tune_pid.py --n-iter 500       # more iterations
#   python tune_pid.py --random-only      # force random search