"""
CMA-ES tuning — replicates run.py exactly.

Runs all 4 targets from targets.csv consecutively (as the professor grades),
20 s per target, check_action clip at ±1 m/s, scores mean+std of pos error
over the last 10 s of each target window.

Run:
    pip install cma
    python tune_pid.py
    python tune_pid.py --n-iter 300   # more evaluations
"""

import argparse
import json
import math
import numpy as np
import pybullet as p
import pybullet_data

from src.PID_controller import PIDController
from src.tello_controller import TelloController


# ---------------------------------------------------------------------------
# Must match run.py exactly
# ---------------------------------------------------------------------------
SIM_HZ          = 1000
CTRL_HZ         = 50
SECS_PER_TARGET = 20.0    # professor gives 20 s per target
STEADY_SECONDS  = 10.0    # last 10 s is the measurement window
VEL_CLIP        = 1.0     # check_action clips to ±1 m/s
YAW_CLIP        = 1.74533 # check_action clips yaw rate to ±1.74533 rad/s

# Exact targets from targets.csv (z < 0 rows skipped as run.py does)
TARGETS = [
    (2.0,  2.0, 2.0, 0.0  ),
    (-2.0, 2.0, 2.0, 1.57 ),
    (-2.0,-2.0, 2.0, 3.14 ),
    (2.0, -2.0, 2.0, 4.71 ),
]

DRONE_URDF = "resources/tello.urdf"
START_POS  = [0, 0, 1]

M       = 0.088
L       = 0.06
KF      = 0.566e-5
KM      = 0.762e-7
K_TRANS = np.array([3.365e-2, 3.365e-2, 3.365e-2])
TM      = 0.0163

# CMA-ES search space  [kp, kd, kp_yaw, ki_yaw, kd_yaw]
SPACE_LB = np.array([0.5,  0.0,  0.5,  0.0,  0.0])
SPACE_UB = np.array([5.0,  2.0,  5.0,  0.1,  0.5])


# ---------------------------------------------------------------------------
# Full run: all 4 targets consecutively, PID state preserved across targets
# ---------------------------------------------------------------------------
def run_trial(kp, kd, kp_yaw, ki_yaw, kd_yaw, verbose=False):
    physics_client = p.connect(p.DIRECT)
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

    timestep       = 1.0 / SIM_HZ
    ctrl_dt        = 1.0 / CTRL_HZ
    steps_per_ctrl = int(round(ctrl_dt / timestep))
    steps_per_tgt  = int(SECS_PER_TARGET * SIM_HZ)
    steady_start   = steps_per_tgt - int(STEADY_SECONDS * SIM_HZ)

    prev_rpm    = np.zeros(4)
    desired_vel = np.zeros(3)
    yaw_rate_sp = 0.0
    last_target = None

    all_scores = []

    for tgt_x, tgt_y, tgt_z, tgt_yaw in TARGETS:
        pos_errors = []
        loop_counter = 0

        for step in range(steps_per_tgt):
            pos, quat = p.getBasePositionAndOrientation(drone_id,
                                                        physicsClientId=physics_client)
            lv_world, av_world = p.getBaseVelocity(drone_id,
                                                   physicsClientId=physics_client)

            roll, pitch, yaw = p.getEulerFromQuaternion(quat)
            yq = p.getQuaternionFromEuler([0, 0, yaw])
            _, inv_yq = p.invertTransform([0, 0, 0], yq)
            _, inv_q  = p.invertTransform([0, 0, 0], quat)
            lin_vel = np.array(p.rotateVector(inv_yq, lv_world))
            ang_vel = np.array(p.rotateVector(inv_q,  av_world))

            loop_counter += 1
            if loop_counter >= steps_per_ctrl:
                loop_counter = 0
                cur_pos   = np.array(pos)
                target    = (tgt_x, tgt_y, tgt_z, tgt_yaw)

                # Reset derivative on target change (matches controller.py fix)
                if last_target != target:
                    pos_pid.previous_error = np.array([0.0, 0.0, 0.0])
                    last_target = target

                pos_error = np.array([tgt_x - cur_pos[0],
                                      tgt_y - cur_pos[1],
                                      tgt_z - cur_pos[2]])

                vel_world = pos_pid.control_update(pos_error, ctrl_dt)

                yaw_error = tgt_yaw - yaw
                yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
                yaw_rate_sp = float(np.clip(
                    yaw_pid.control_update(np.array([yaw_error, 0, 0]), ctrl_dt)[0],
                    -YAW_CLIP, YAW_CLIP,
                ))

                vx_b = vel_world[0] * math.cos(yaw) + vel_world[1] * math.sin(yaw)
                vy_b = -vel_world[0] * math.sin(yaw) + vel_world[1] * math.cos(yaw)
                # check_action clip — exactly as run.py does
                desired_vel = np.array([
                    np.clip(vx_b,         -VEL_CLIP, VEL_CLIP),
                    np.clip(vy_b,         -VEL_CLIP, VEL_CLIP),
                    np.clip(vel_world[2], -VEL_CLIP, VEL_CLIP),
                ])

                if step >= steady_start:
                    pos_errors.append(np.linalg.norm(pos_error))

            rpm = tello.compute_control(desired_vel, lin_vel, quat, ang_vel,
                                        yaw_rate_sp, timestep)
            rpm_d = (rpm - prev_rpm) / TM
            rpm = prev_rpm + rpm_d * timestep
            prev_rpm = rpm

            rotation     = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            omega        = rpm * (2 * np.pi / 60)
            motor_forces = omega**2 * KF
            thrust       = np.array([0, 0, np.sum(motor_forces)])
            vel_body     = rotation.T @ lv_world
            force        = -K_TRANS * vel_body + thrust

            z_t = omega**2 * KM
            torques = np.array([
                (-motor_forces[0] + motor_forces[1] + motor_forces[2] - motor_forces[3]) * L,
                (-motor_forces[0] + motor_forces[1] - motor_forces[2] + motor_forces[3]) * L,
                -z_t[0] - z_t[1] + z_t[2] + z_t[3],
            ])

            p.applyExternalForce(drone_id, -1, force.tolist(), [0, 0, 0],
                                 p.LINK_FRAME, physicsClientId=physics_client)
            p.applyExternalTorque(drone_id, -1, torques.tolist(),
                                  p.LINK_FRAME, physicsClientId=physics_client)
            p.stepSimulation(physicsClientId=physics_client)

        arr  = np.array(pos_errors) if pos_errors else np.array([9999.0])
        mean = float(np.mean(arr))
        std  = float(np.std(arr))
        score = mean + std
        all_scores.append(score)
        if verbose:
            print(f"  ({tgt_x:+.0f},{tgt_y:+.0f},{tgt_z:.0f},yaw={tgt_yaw:.2f})"
                  f"  mean={mean*100:.2f}cm  std={std*100:.2f}cm  score={score:.4f}")

    p.disconnect(physics_client)
    return float(np.mean(all_scores))


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
_trial_count = [0]

def objective(params):
    kp, kd, kp_yaw, ki_yaw, kd_yaw = [float(x) for x in params]
    _trial_count[0] += 1
    print(f"Trial {_trial_count[0]:3d}  "
          f"kp={kp:.4f} kd={kd:.4f}  "
          f"kp_yaw={kp_yaw:.3f} ki_yaw={ki_yaw:.4f} kd_yaw={kd_yaw:.3f}",
          end="  ", flush=True)
    score = run_trial(kp, kd, kp_yaw, ki_yaw, kd_yaw)
    print(f"→ {score:.4f}")
    return score


def print_results(best_params, best_score):
    kp, kd, kp_yaw, ki_yaw, kd_yaw = [float(x) for x in best_params]
    print("\n" + "=" * 55)
    print(f"BEST (score={best_score:.4f}):")
    print(f"  Pos  Kp={kp:.4f}  Ki=0.0  Kd={kd:.4f}")
    print(f"  Yaw  Kp={kp_yaw:.4f}  Ki={ki_yaw:.4f}  Kd={kd_yaw:.4f}")
    print("=" * 55)
    print("\nPaste into controller.py:")
    print(f"Kp_pos = np.array([{kp:.4f}, {kp:.4f}, {kp:.4f}])")
    print(f"Ki_pos = np.array([0.0000, 0.0000, 0.0000])")
    print(f"Kd_pos = np.array([{kd:.4f}, {kd:.4f}, {kd:.4f}])")
    print(f"Kp_yaw = np.array([{kp_yaw:.4f}, 0.0, 0.0])")
    print(f"Ki_yaw = np.array([{ki_yaw:.4f}, 0.0, 0.0])")
    print(f"Kd_yaw = np.array([{kd_yaw:.4f}, 0.0, 0.0])")
    with open("best_gains.json", "w") as f:
        json.dump({"kp": kp, "kd": kd, "kp_yaw": kp_yaw,
                   "ki_yaw": ki_yaw, "kd_yaw": kd_yaw, "score": best_score}, f, indent=2)
    print("\nSaved to best_gains.json\n\nVerifying:")
    run_trial(kp, kd, kp_yaw, ki_yaw, kd_yaw, verbose=True)


# ---------------------------------------------------------------------------
# CMA-ES optimiser
# ---------------------------------------------------------------------------
def tune_cmaes(n_iter):
    import cma
    from multiprocessing import Pool, cpu_count

    x0     = [1.72, 0.40, 1.14, 0.028, 0.18]
    sigma0 = 0.4

    opts = cma.CMAOptions()
    opts["maxfevals"] = n_iter
    opts["bounds"]    = [SPACE_LB.tolist(), SPACE_UB.tolist()]
    opts["tolx"]      = 1e-4
    opts["tolfun"]    = 1e-5
    opts["verbose"]   = -9

    n_workers = max(1, cpu_count() - 1)
    print(f"CMA-ES  (max {n_iter} evals, {n_workers} workers)")
    print("=" * 55)

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    with Pool(n_workers) as pool:
        while not es.stop():
            solutions = es.ask()
            clipped   = [np.clip(x, SPACE_LB, SPACE_UB).tolist() for x in solutions]
            fitnesses = pool.map(objective, clipped)
            es.tell(solutions, fitnesses)

    best = np.clip(es.result.xbest, SPACE_LB, SPACE_UB).tolist()
    return best, float(es.result.fbest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter", type=int, default=200)
    args = parser.parse_args()

    try:
        best_params, best_score = tune_cmaes(args.n_iter)
    except ImportError:
        print("cma not found — pip install cma")
        return

    print_results(best_params, best_score)


if __name__ == "__main__":
    main()

# USAGE:
#   pip install cma
#   python tune_pid.py
#   python tune_pid.py --n-iter 300
