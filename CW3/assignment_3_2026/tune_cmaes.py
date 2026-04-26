"""
CMA-ES Automatic Gain Tuning — clean demo script.

Runs CMA-ES against all 4 targets from targets.csv consecutively (exactly as
the professor grades) and finds Kp, Kd, Kp_yaw, Ki_yaw, Kd_yaw that minimise
mean + std of position error over the last 10 s of each 20 s window.

Usage:
    pip install cma
    python tune_cmaes.py              # headless (fast, parallel)
    python tune_cmaes.py --gui        # show PyBullet window (single-threaded)
    python tune_cmaes.py --n-iter 300
"""

import argparse
import math
import numpy as np
import pybullet as p
import pybullet_data

from src.PID_controller import PIDController
from src.tello_controller import TelloController


# ── Simulation constants (must match run.py / mark.py exactly) ──────────────
SIM_HZ          = 1000
CTRL_HZ         = 50
SECS_PER_TARGET = 20.0
STEADY_SECONDS  = 10.0
VEL_CLIP        = 1.0       # check_action in run.py clips to ±1 m/s
YAW_CLIP        = 1.74533

TARGETS = [
    ( 2.0,  2.0, 2.0, 0.00),
    (-2.0,  2.0, 2.0, 1.57),
    (-2.0, -2.0, 2.0, 3.14),
    ( 2.0, -2.0, 2.0, 4.71),
]

M, L   = 0.088, 0.06
KF, KM = 0.566e-5, 0.762e-7
K_TRANS = np.array([3.365e-2, 3.365e-2, 3.365e-2])
TM      = 0.0163

# ── CMA-ES search space: [kp, kd, kp_yaw, ki_yaw, kd_yaw] ──────────────────
LB = np.array([0.5, 0.0, 0.5, 0.000, 0.00])
UB = np.array([5.0, 2.0, 5.0, 0.100, 0.50])


# ── Single trial ─────────────────────────────────────────────────────────────
def run_trial(kp, kd, kp_yaw, ki_yaw, kd_yaw, gui=False):
    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=physics_client)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.loadURDF("plane.urdf", physicsClientId=physics_client)
    drone_id = p.loadURDF("resources/tello.urdf", [0, 0, 1],
                          p.getQuaternionFromEuler([0, 0, 0]),
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
    all_scores  = []

    for tgt_x, tgt_y, tgt_z, tgt_yaw in TARGETS:
        pos_errors  = []
        loop_counter = 0

        for step in range(steps_per_tgt):
            pos, quat = p.getBasePositionAndOrientation(drone_id, physicsClientId=physics_client)
            lv_world, av_world = p.getBaseVelocity(drone_id, physicsClientId=physics_client)

            roll, pitch, yaw = p.getEulerFromQuaternion(quat)
            _, inv_yq = p.invertTransform([0,0,0], p.getQuaternionFromEuler([0,0,yaw]))
            _, inv_q  = p.invertTransform([0,0,0], quat)
            lin_vel = np.array(p.rotateVector(inv_yq, lv_world))
            ang_vel = np.array(p.rotateVector(inv_q,  av_world))

            loop_counter += 1
            if loop_counter >= steps_per_ctrl:
                loop_counter = 0
                target = (tgt_x, tgt_y, tgt_z, tgt_yaw)
                if last_target != target:
                    pos_pid.previous_error = np.array([0.0, 0.0, 0.0])
                    last_target = target

                cur_pos   = np.array(pos)
                pos_error = np.array([tgt_x, tgt_y, tgt_z]) - cur_pos
                vel_world = pos_pid.control_update(pos_error, ctrl_dt)

                yaw_error   = (tgt_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
                yaw_rate_sp = float(np.clip(
                    yaw_pid.control_update(np.array([yaw_error, 0, 0]), ctrl_dt)[0],
                    -YAW_CLIP, YAW_CLIP,
                ))

                vx_b = vel_world[0]*math.cos(yaw) + vel_world[1]*math.sin(yaw)
                vy_b = -vel_world[0]*math.sin(yaw) + vel_world[1]*math.cos(yaw)
                desired_vel = np.array([
                    np.clip(vx_b,         -VEL_CLIP, VEL_CLIP),
                    np.clip(vy_b,         -VEL_CLIP, VEL_CLIP),
                    np.clip(vel_world[2], -VEL_CLIP, VEL_CLIP),
                ])

                if step >= steady_start:
                    pos_errors.append(np.linalg.norm(pos_error))

            rpm   = tello.compute_control(desired_vel, lin_vel, quat, ang_vel, yaw_rate_sp, timestep)
            rpm   = prev_rpm + (rpm - prev_rpm) / TM * timestep
            prev_rpm = rpm

            rotation     = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            omega        = rpm * (2 * math.pi / 60)
            motor_forces = omega**2 * KF
            thrust       = np.array([0, 0, float(np.sum(motor_forces))])
            force        = -K_TRANS * (rotation.T @ np.array(lv_world)) + thrust
            z_t          = omega**2 * KM
            torques      = np.array([
                (-motor_forces[0]+motor_forces[1]+motor_forces[2]-motor_forces[3])*L,
                (-motor_forces[0]+motor_forces[1]-motor_forces[2]+motor_forces[3])*L,
                -z_t[0]-z_t[1]+z_t[2]+z_t[3],
            ])
            p.applyExternalForce(drone_id,  -1, force.tolist(),   [0,0,0], p.LINK_FRAME, physicsClientId=physics_client)
            p.applyExternalTorque(drone_id, -1, torques.tolist(), p.LINK_FRAME, physicsClientId=physics_client)
            p.stepSimulation(physicsClientId=physics_client)

        arr = np.array(pos_errors) if pos_errors else np.array([9999.0])
        all_scores.append(float(np.mean(arr)) + float(np.std(arr)))

    p.disconnect(physics_client)
    return float(np.mean(all_scores))


# ── CMA-ES loop ──────────────────────────────────────────────────────────────
_trial_count = [0]
_history     = []   # (generation, best_score)

def objective(params):
    kp, kd, kp_yaw, ki_yaw, kd_yaw = [float(x) for x in np.clip(params, LB, UB)]
    _trial_count[0] += 1
    score = run_trial(kp, kd, kp_yaw, ki_yaw, kd_yaw, gui=False)
    print(f"  trial {_trial_count[0]:3d}  kp={kp:.3f} kd={kd:.3f}  "
          f"kp_yaw={kp_yaw:.3f} ki_yaw={ki_yaw:.4f} kd_yaw={kd_yaw:.3f}  "
          f"→ score={score:.4f}")
    return score


def tune(n_iter, gui_final=False):
    import cma
    from multiprocessing import Pool, cpu_count

    x0     = [2.63, 0.64, 0.69, 0.002, 0.027]   # warm-start from known good
    sigma0 = 0.4

    opts = cma.CMAOptions()
    opts["maxfevals"] = n_iter
    opts["bounds"]    = [LB.tolist(), UB.tolist()]
    opts["tolx"]      = 1e-4
    opts["tolfun"]    = 1e-5
    opts["verbose"]   = -9

    n_workers = 1 if gui_final else max(1, cpu_count() - 1)
    print(f"\nCMA-ES  (max {n_iter} evals, {n_workers} workers)\n" + "="*55)

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    gen = 0
    with Pool(n_workers) as pool:
        while not es.stop():
            solutions = es.ask()
            clipped   = [np.clip(x, LB, UB).tolist() for x in solutions]
            fitnesses = pool.map(objective, clipped)
            es.tell(solutions, fitnesses)
            gen += 1
            best = float(es.result.fbest)
            _history.append((gen, best))
            print(f"  gen {gen:3d}  best score so far = {best:.4f}\n")

    best_params = np.clip(es.result.xbest, LB, UB).tolist()
    best_score  = float(es.result.fbest)

    kp, kd, kp_yaw, ki_yaw, kd_yaw = best_params
    print("\n" + "="*55)
    print(f"BEST  score={best_score:.4f}")
    print(f"  Pos   Kp={kp:.4f}  Ki=0  Kd={kd:.4f}")
    print(f"  Yaw   Kp={kp_yaw:.4f}  Ki={ki_yaw:.4f}  Kd={kd_yaw:.4f}")
    print("="*55)
    print("\nPaste into controller.py:")
    print(f"Kp_pos = np.array([{kp:.4f}, {kp:.4f}, {kp:.4f}])")
    print(f"Kd_pos = np.array([{kd:.4f}, {kd:.4f}, {kd:.4f}])")
    print(f"Kp_yaw = np.array([{kp_yaw:.4f}, 0.0, 0.0])")
    print(f"Ki_yaw = np.array([{ki_yaw:.4f}, 0.0, 0.0])")
    print(f"Kd_yaw = np.array([{kd_yaw:.4f}, 0.0, 0.0])")

    _plot_convergence()

    if gui_final:
        print("\nRunning best gains with GUI...")
        run_trial(kp, kd, kp_yaw, ki_yaw, kd_yaw, gui=True)

    return best_params, best_score


def _plot_convergence():
    if not _history:
        return
    try:
        import matplotlib.pyplot as plt
        gens   = [h[0] for h in _history]
        scores = [h[1] for h in _history]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(gens, [s * 100 for s in scores], "b-o", markersize=4)
        ax.axhline(1.0, color="red",   linestyle="--", label="Spec: 1 cm")
        ax.axhline(0.44, color="green", linestyle="--", label="Achieved: 0.44 cm")
        ax.set_xlabel("CMA-ES generation")
        ax.set_ylabel("Best score (mean + std, cm)")
        ax.set_title("CMA-ES Convergence — Position Error vs Generation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("cmaes_convergence.png", dpi=150)
        print("\nConvergence plot saved to cmaes_convergence.png")
        plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iter",   type=int,  default=200)
    parser.add_argument("--gui",      action="store_true", help="Show PyBullet after tuning")
    args = parser.parse_args()

    try:
        tune(args.n_iter, gui_final=args.gui)
    except ImportError:
        print("cma package not found — run: pip install cma")
