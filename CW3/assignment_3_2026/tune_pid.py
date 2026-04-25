"""
Bayesian optimisation of position PID gains (pure PD, no DOB).

Matches controller.py exactly: PD outer loop, VEL_LIMIT=3.5, no integral, no DOB.
Scores mean+std of position error over the last 10 s — the professor's marking window.

Run from the assignment_3_2026 directory:
    python tune_pid.py                 # Bayesian Opt (scikit-optimize GP+EI)
    python tune_pid.py --n-iter 50     # fewer calls (default 60)
    python tune_pid.py --nelder-mead   # fallback to Nelder-Mead (no skopt needed)

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
# Simulation parameters — identical to professor's marking setup
# ---------------------------------------------------------------------------
SIM_HZ         = 1000
CTRL_HZ        = 50
SIM_SECONDS    = 20.0   # 10 s approach + 10 s scoring
STEADY_SECONDS = 10.0   # score the last 10 s
VEL_LIMIT      = 3.5
YAW_LIMIT      = 1.5

# Three representative targets — far reach, 3-D diagonal, short+yaw
EVAL_TARGETS = [
    (4.0,  0.0,  1.0, 0.0 ),
    (2.0,  2.0,  2.0, 0.0 ),
    (1.0, -1.0,  0.5, 1.57),
]

DRONE_URDF = "resources/tello.urdf"
START_POS  = [0, 0, 1]

M       = 0.088
L       = 0.06
KF      = 0.566e-5
KM      = 0.762e-7
K_TRANS = np.array([3.365e-2, 3.365e-2, 3.365e-2])
TM      = 0.0163

# Search space  [kp,  kd,  kp_yaw, ki_yaw, kd_yaw]
# Kd upper bound < 1.0: guarantees net-positive velocity at saturation exit
#   (vel_cmd = Kp*e - Kd*VEL_LIMIT > 0  iff  Kp*e > Kd*VEL_LIMIT at exit point)
SPACE_LB = np.array([0.6,  0.0,  0.5,  0.0,  0.0])
SPACE_UB = np.array([3.0,  0.95, 5.0,  0.05, 0.5])


# ---------------------------------------------------------------------------
# One headless simulation trial  (pure PD, matches controller.py)
# ---------------------------------------------------------------------------
def run_trial(kp, kd, kp_yaw, ki_yaw, kd_yaw,
              verbose=False, gui=False, target_override=None,
              wind_enabled=True, wind_seed=42):
    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=physics_client)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.loadURDF("plane.urdf", physicsClientId=physics_client)

    start_orn = p.getQuaternionFromEuler([0, 0, 0])
    drone_id  = p.loadURDF(DRONE_URDF, START_POS, start_orn,
                           physicsClientId=physics_client)

    pos_pid = PIDController(
        Kp=np.array([kp, kp, kp]),
        Ki=np.array([0.0, 0.0, 0.0]),
        Kd=np.array([kd, kd, kd]),
        Ki_sat=np.array([1.0, 1.0, 1.0]),
    )
    yaw_pid = PIDController(
        Kp=np.array([kp_yaw, 0.0, 0.0]),
        Ki=np.array([ki_yaw, 0.0, 0.0]),
        Kd=np.array([kd_yaw, 0.0, 0.0]),
        Ki_sat=np.array([0.5, 0.0, 0.0]),
    )
    tello = TelloController(9.81, M, L, 0.35, KF, KM)

    import random as _random
    _random.seed(wind_seed)
    wind_sim = Wind(max_steady_state=0.02, max_gust=0.02, k_gusts=0.1) if wind_enabled else None

    timestep       = 1.0 / SIM_HZ
    ctrl_timestep  = 1.0 / CTRL_HZ
    steps_per_ctrl = int(round(ctrl_timestep / timestep))
    total_steps    = int(SIM_SECONDS * SIM_HZ)
    steady_start   = total_steps - int(STEADY_SECONDS * SIM_HZ)

    prev_rpm    = np.zeros(4)
    desired_vel = np.zeros(3)
    yaw_rate_sp = 0.0
    loop_counter = 0
    pos_errors  = []
    yaw_errors  = []

    tgt_x, tgt_y, tgt_z, tgt_yaw = target_override if target_override is not None else EVAL_TARGETS[0]

    for step in range(total_steps):
        pos, quat = p.getBasePositionAndOrientation(drone_id, physicsClientId=physics_client)
        lin_vel_world, ang_vel_world = p.getBaseVelocity(drone_id, physicsClientId=physics_client)

        roll, pitch, yaw = p.getEulerFromQuaternion(quat)
        yaw_quat = p.getQuaternionFromEuler([0, 0, yaw])
        _, inv_quat_yaw = p.invertTransform([0, 0, 0], yaw_quat)
        _, inv_quat     = p.invertTransform([0, 0, 0], quat)

        lin_vel = np.array(p.rotateVector(inv_quat_yaw, lin_vel_world))
        ang_vel = np.array(p.rotateVector(inv_quat,     ang_vel_world))

        loop_counter += 1
        if loop_counter >= steps_per_ctrl:
            loop_counter = 0

            cur_pos   = np.array(pos)
            pos_error = np.array([tgt_x - cur_pos[0],
                                  tgt_y - cur_pos[1],
                                  tgt_z - cur_pos[2]])

            vel_world = pos_pid.control_update(pos_error, ctrl_timestep)

            yaw_error = tgt_yaw - yaw
            yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
            yaw_rate_sp = float(np.clip(
                yaw_pid.control_update(np.array([yaw_error, 0, 0]), ctrl_timestep)[0],
                -YAW_LIMIT, YAW_LIMIT,
            ))

            vx_b = vel_world[0] * math.cos(yaw) + vel_world[1] * math.sin(yaw)
            vy_b = -vel_world[0] * math.sin(yaw) + vel_world[1] * math.cos(yaw)
            desired_vel = np.array([
                np.clip(vx_b,          -VEL_LIMIT, VEL_LIMIT),
                np.clip(vy_b,          -VEL_LIMIT, VEL_LIMIT),
                np.clip(vel_world[2],  -VEL_LIMIT, VEL_LIMIT),
            ])

            if step >= steady_start:
                pos_errors.append(np.linalg.norm(pos_error))
                yaw_errors.append(abs(yaw_error))

        rpm = tello.compute_control(desired_vel, lin_vel, quat, ang_vel, yaw_rate_sp, timestep)
        rpm_deriv = (rpm - prev_rpm) / TM
        rpm = prev_rpm + rpm_deriv * timestep
        prev_rpm = rpm

        rotation     = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        omega        = rpm * (2 * np.pi / 60)
        motor_forces = omega**2 * KF
        thrust       = np.array([0, 0, np.sum(motor_forces)])
        vel_body     = rotation.T @ lin_vel_world
        force        = -K_TRANS * vel_body + thrust

        z_torques = omega**2 * KM
        z_torque  = -z_torques[0] - z_torques[1] + z_torques[2] + z_torques[3]
        x_torque  = (-motor_forces[0] + motor_forces[1] + motor_forces[2] - motor_forces[3]) * L
        y_torque  = (-motor_forces[0] + motor_forces[1] - motor_forces[2] + motor_forces[3]) * L
        torques   = np.array([x_torque, y_torque, z_torque])

        p.applyExternalForce(drone_id, -1, force.tolist(), [0,0,0],
                             p.LINK_FRAME, physicsClientId=physics_client)
        p.applyExternalTorque(drone_id, -1, torques.tolist(),
                              p.LINK_FRAME, physicsClientId=physics_client)
        if wind_sim is not None:
            wind_force = wind_sim.get_wind(timestep)
            p.applyExternalForce(drone_id, -1, wind_force.tolist(), list(pos),
                                 p.WORLD_FRAME, physicsClientId=physics_client)
        p.stepSimulation(physicsClientId=physics_client)

    p.disconnect(physics_client)

    arr = np.array(pos_errors) if pos_errors else np.array([9999.0])
    mean_pos = float(np.mean(arr))
    std_pos  = float(np.std(arr))
    mean_yaw = float(np.mean(yaw_errors)) if yaw_errors else 9999.0
    # Score = mean + std of pos error (exactly the spec) + small yaw penalty
    score = mean_pos + std_pos + 0.3 * mean_yaw
    if verbose:
        print(f"  pos mean={mean_pos:.4f} std={std_pos:.4f}  yaw={mean_yaw:.4f}  score={score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Objective: average score across all eval targets
# ---------------------------------------------------------------------------
_trial_count = [0]

def objective(params):
    kp, kd, kp_yaw, ki_yaw, kd_yaw = params
    _trial_count[0] += 1
    print(f"Trial {_trial_count[0]:3d}  kp={kp:.4f} kd={kd:.4f}  "
          f"kp_yaw={kp_yaw:.3f} ki_yaw={ki_yaw:.4f} kd_yaw={kd_yaw:.3f}", end="  ")
    scores = [
        run_trial(kp, kd, kp_yaw, ki_yaw, kd_yaw,
                  wind_enabled=True, wind_seed=42 + i, target_override=tgt)
        for i, tgt in enumerate(EVAL_TARGETS)
    ]
    score = float(np.mean(scores))
    print(f"→ {score:.4f}  {[f'{s:.4f}' for s in scores]}")
    return score


def print_results(best_params, best_score):
    kp, kd, kp_yaw, ki_yaw, kd_yaw = best_params
    print("\n" + "=" * 55)
    print(f"BEST gains (score={best_score:.4f}):")
    print(f"  Pos  Kp={kp:.4f}  Ki=0.0000  Kd={kd:.4f}")
    print(f"  Yaw  Kp={kp_yaw:.4f}  Ki={ki_yaw:.4f}  Kd={kd_yaw:.4f}")
    print("=" * 55)
    print("\nPaste into controller.py:")
    print(f"Kp_pos = np.array([{kp:.4f}, {kp:.4f}, {kp:.4f}])")
    print(f"Ki_pos = np.array([0.0000, 0.0000, 0.0000])")
    print(f"Kd_pos = np.array([{kd:.4f}, {kd:.4f}, {kd:.4f}])")
    print(f"Kp_yaw = np.array([{kp_yaw:.4f}, 0.0, 0.0])")
    print(f"Ki_yaw = np.array([{ki_yaw:.4f}, 0.0, 0.0])")
    print(f"Kd_yaw = np.array([{kd_yaw:.4f}, 0.0, 0.0])")
    gains = {"kp": kp, "kd": kd, "kp_yaw": kp_yaw, "ki_yaw": ki_yaw,
             "kd_yaw": kd_yaw, "score": best_score}
    with open("best_gains.json", "w") as f:
        json.dump(gains, f, indent=2)
    print("\nSaved to best_gains.json")
    print("\nVerifying across all targets:")
    for i, tgt in enumerate(EVAL_TARGETS):
        print(f"  Target {i+1} {tgt}:", end="")
        run_trial(kp, kd, kp_yaw, ki_yaw, kd_yaw,
                  wind_enabled=True, wind_seed=42 + i,
                  verbose=True, target_override=tgt)


# ---------------------------------------------------------------------------
# Optimisers
# ---------------------------------------------------------------------------
def tune_bayesian(n_calls):
    """Gaussian Process + Expected Improvement (scikit-optimize)."""
    from skopt import gp_minimize
    from skopt.space import Real

    space = [
        Real(float(SPACE_LB[0]), float(SPACE_UB[0]), name="kp"),
        Real(float(SPACE_LB[1]), float(SPACE_UB[1]), name="kd"),
        Real(float(SPACE_LB[2]), float(SPACE_UB[2]), name="kp_yaw"),
        Real(float(SPACE_LB[3]), float(SPACE_UB[3]), name="ki_yaw"),
        Real(float(SPACE_LB[4]), float(SPACE_UB[4]), name="kd_yaw"),
    ]
    print("=" * 55)
    print(f"Bayesian Optimisation — GP + EI  ({n_calls} calls)")
    print("=" * 55)
    result = gp_minimize(
        objective, space,
        n_calls=n_calls, n_initial_points=10,
        acq_func="EI", noise=1e-10, random_state=42,
        x0=[1.8124, 0.8, 2.5, 0.01, 0.05],   # warm-start from previous best
    )
    return result.x, result.fun


def tune_nelder_mead(n_iter):
    """Nelder-Mead fallback — no scikit-optimize needed."""
    from scipy.optimize import minimize
    print("=" * 55)
    print(f"Nelder-Mead simplex ({n_iter} iterations)")
    print("=" * 55)
    x0 = np.array([1.8124, 0.8, 2.5, 0.01, 0.05])
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": n_iter, "xatol": 1e-3, "fatol": 1e-4,
                            "disp": True, "adaptive": True})
    params = np.clip(res.x, SPACE_LB, SPACE_UB)
    return params.tolist(), float(res.fun)


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian Optimisation of Tello PD position gains.")
    parser.add_argument("--n-iter", type=int, default=60,
                        help="BO calls / NM iterations (default 60)")
    parser.add_argument("--nelder-mead", action="store_true",
                        help="Use Nelder-Mead instead of Bayesian Opt")
    args = parser.parse_args()

    if args.nelder_mead:
        best_params, best_score = tune_nelder_mead(args.n_iter)
    else:
        try:
            best_params, best_score = tune_bayesian(args.n_iter)
        except ImportError:
            print("scikit-optimize not found — install with: pip install scikit-optimize")
            print("Falling back to Nelder-Mead")
            best_params, best_score = tune_nelder_mead(args.n_iter)

    print_results(best_params, best_score)


if __name__ == "__main__":
    main()

# USAGE:
#   pip install scikit-optimize           # one-time install
#   python tune_pid.py                    # BO with 60 evaluations
#   python tune_pid.py --n-iter 100       # more evaluations
#   python tune_pid.py --nelder-mead      # no scikit-optimize needed