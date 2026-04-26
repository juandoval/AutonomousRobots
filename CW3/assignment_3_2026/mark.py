"""
Automated marking simulation — replicates the professor's grading exactly.

Runs all targets from targets.csv, 20 s of simulation time each,
scores mean + std of position error over the last 10 s per target.
No manual key presses needed.

Usage:
    python mark.py          # headless (fast)
    python mark.py --gui    # with PyBullet window
"""

import argparse
import csv
import importlib
import math
import numpy as np
import pybullet as p
import pybullet_data
import time

import controller as ctrl
from src.tello_controller import TelloController


SIM_HZ          = 1000
CTRL_HZ         = 50
SECS_PER_TARGET = 20.0
STEADY_SECONDS  = 10.0
SPEC_MEAN       = 0.01   # m
SPEC_STD        = 0.01   # m

M       = 0.088
L       = 0.06
KF      = 0.566e-5
KM      = 0.762e-7
K_TRANS = np.array([3.365e-2, 3.365e-2, 3.365e-2])
K_ROT   = np.array([4.609e-3, 4.609e-3, 4.609e-3])
TM      = 0.0163


def load_targets():
    targets = []
    try:
        with open("targets.csv") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) != 4:
                    continue
                if float(row[2]) < 0:   # run.py skips z < 0
                    continue
                targets.append(tuple(float(x) for x in row))
    except FileNotFoundError:
        pass
    return targets or [(0.0, 0.0, 0.0, 0.0)]


def run_marking(gui=False):
    importlib.reload(ctrl)   # fresh controller state each run

    targets = load_targets()

    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=physics_client)
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    p.loadURDF("plane.urdf", physicsClientId=physics_client)

    start_pos = [0, 0, 1]
    start_orn = p.getQuaternionFromEuler([0, 0, 0])
    drone_id  = p.loadURDF("resources/tello.urdf", start_pos, start_orn,
                           physicsClientId=physics_client)

    tello = TelloController(9.81, M, L, 0.35, KF, KM)

    timestep       = 1.0 / SIM_HZ
    ctrl_dt        = 1.0 / CTRL_HZ
    steps_per_ctrl = int(round(ctrl_dt / timestep))
    steps_per_tgt  = int(SECS_PER_TARGET * SIM_HZ)
    steady_start   = steps_per_tgt - int(STEADY_SECONDS * SIM_HZ)

    prev_rpm    = np.zeros(4)
    desired_vel = np.zeros(3)
    yaw_rate_sp = 0.0

    results = []
    # Per-target timeseries for plotting: list of dicts
    timeseries = []

    print(f"\nMarking {len(targets)} target(s), {SECS_PER_TARGET:.0f}s each "
          f"({STEADY_SECONDS:.0f}s scoring window)\n")

    for tgt_idx, target in enumerate(targets):
        tgt_x, tgt_y, tgt_z, tgt_yaw = target
        pos_errors = []
        yaw_errors = []
        loop_counter = 0

        # Timeseries storage (at control rate)
        ts_time     = []
        ts_pos_err  = []
        ts_yaw_err  = []

        print(f"Target {tgt_idx+1}/{len(targets)}: "
              f"({tgt_x:+.1f}, {tgt_y:+.1f}, {tgt_z:.1f}, yaw={tgt_yaw:.2f} rad) ...",
              end=" ", flush=True)

        for step in range(steps_per_tgt):
            pos, quat = p.getBasePositionAndOrientation(drone_id,
                                                        physicsClientId=physics_client)
            lv_world, av_world = p.getBaseVelocity(drone_id,
                                                   physicsClientId=physics_client)

            yaw_q = p.getQuaternionFromEuler([0, 0, p.getEulerFromQuaternion(quat)[2]])
            _, inv_yq = p.invertTransform([0, 0, 0], yaw_q)
            _, inv_q  = p.invertTransform([0, 0, 0], quat)
            lin_vel = np.array(p.rotateVector(inv_yq, lv_world))
            ang_vel = np.array(p.rotateVector(inv_q,  av_world))

            loop_counter += 1
            if loop_counter >= steps_per_ctrl:
                loop_counter = 0
                state = np.concatenate((pos, p.getEulerFromQuaternion(quat)))
                raw = ctrl.controller(state, target, ctrl_dt, wind_enabled=False)

                # Exact same check_action as run.py
                desired_vel = np.array([
                    np.clip(raw[0], -1.0, 1.0),
                    np.clip(raw[1], -1.0, 1.0),
                    np.clip(raw[2], -1.0, 1.0),
                ])
                yaw_rate_sp = float(np.clip(raw[3], -1.74533, 1.74533))

                cur_pos  = np.array(pos)
                pe       = float(np.linalg.norm(np.array([tgt_x, tgt_y, tgt_z]) - cur_pos))
                ye       = float(abs((tgt_yaw - p.getEulerFromQuaternion(quat)[2] + math.pi)
                                     % (2 * math.pi) - math.pi))
                t_sim    = step / SIM_HZ

                ts_time.append(t_sim)
                ts_pos_err.append(pe)
                ts_yaw_err.append(ye)

                if step >= steady_start:
                    pos_errors.append(pe)
                    yaw_errors.append(ye)

            # Dynamics — mirrors run.py compute_dynamics + motor_model
            rotation     = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            rpm_desired  = tello.compute_control(desired_vel, lin_vel, quat,
                                                 ang_vel, yaw_rate_sp, timestep)
            rpm_d        = (rpm_desired - prev_rpm) / TM
            rpm          = prev_rpm + rpm_d * timestep
            prev_rpm     = rpm

            omega        = rpm * (2 * math.pi / 60)
            motor_forces = omega**2 * KF
            thrust       = np.array([0, 0, float(np.sum(motor_forces))])
            vel_body     = rotation.T @ np.array(lv_world)
            force        = -K_TRANS * vel_body + thrust

            z_t = omega**2 * KM
            torques = np.array([
                (-motor_forces[0]+motor_forces[1]+motor_forces[2]-motor_forces[3])*L,
                (-motor_forces[0]+motor_forces[1]-motor_forces[2]+motor_forces[3])*L,
                -z_t[0]-z_t[1]+z_t[2]+z_t[3],
            ])

            p.applyExternalForce(drone_id, -1, force.tolist(), [0,0,0],
                                 p.LINK_FRAME, physicsClientId=physics_client)
            p.applyExternalTorque(drone_id, -1, torques.tolist(),
                                  p.LINK_FRAME, physicsClientId=physics_client)
            p.stepSimulation(physicsClientId=physics_client)

        arr      = np.array(pos_errors) if pos_errors else np.array([9999.0])
        yaw_arr  = np.array(yaw_errors) if yaw_errors else np.array([9999.0])
        mean_pos = float(np.mean(arr))
        std_pos  = float(np.std(arr))
        mean_yaw = float(np.mean(yaw_arr))
        std_yaw  = float(np.std(yaw_arr))
        pos_pass = mean_pos < SPEC_MEAN and std_pos < SPEC_STD
        yaw_pass = mean_yaw < 0.01 and std_yaw < 0.001
        results.append((mean_pos, std_pos, mean_yaw, std_yaw))
        timeseries.append({
            "target": target,
            "time":   np.array(ts_time),
            "pos":    np.array(ts_pos_err),
            "yaw":    np.array(ts_yaw_err),
            "mean":   mean_pos,
            "std":    std_pos,
            "pass":   pos_pass,
        })

        print(f"pos mean={mean_pos*100:.2f}cm  std={std_pos*100:.2f}cm  "
              f"{'✓' if pos_pass else '✗'}  |  "
              f"yaw mean={mean_yaw:.4f}rad  std={std_yaw:.4f}rad  "
              f"{'✓' if yaw_pass else '✗'}")

    p.disconnect(physics_client)

    # Summary
    means     = [r[0] for r in results]
    stds      = [r[1] for r in results]
    yaw_means = [r[2] for r in results]
    yaw_stds  = [r[3] for r in results]
    overall_mean = float(np.mean(means))
    overall_std  = float(np.mean(stds))
    print(f"\n{'='*55}")
    print(f"Overall  pos mean={overall_mean*100:.2f}cm  "
          f"std={overall_std*100:.2f}cm  "
          f"{'PASS ✓' if overall_mean < SPEC_MEAN and overall_std < SPEC_STD else 'FAIL ✗'}")
    print(f"Spec:    pos mean < {SPEC_MEAN*100:.0f}cm, std < {SPEC_STD*100:.0f}cm")
    print(f"         yaw mean < 0.01 rad, std < 0.001 rad")
    print(f"{'='*55}\n")

    _plot_results(timeseries, overall_mean, overall_std)
    return results


def _plot_results(timeseries, overall_mean, overall_std):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(timeseries)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    steady_start_t = SECS_PER_TARGET - STEADY_SECONDS

    for i, (ax, ts) in enumerate(zip(axes, timeseries)):
        tgt = ts["target"]
        t   = ts["time"]
        pe  = ts["pos"] * 100   # convert to cm

        ax.plot(t, pe, color="#2196F3", linewidth=1.2, label="Position error")

        # Shade scoring window
        ax.axvspan(steady_start_t, SECS_PER_TARGET, alpha=0.12, color="green",
                   label=f"Scoring window ({STEADY_SECONDS:.0f}s)")

        # Mean and spec lines
        mean_cm = ts["mean"] * 100
        ax.axhline(mean_cm,         color="#4CAF50", linewidth=1.5,
                   linestyle="--", label=f"Mean = {mean_cm:.2f} cm")
        ax.axhline(SPEC_MEAN * 100, color="red",     linewidth=1.0,
                   linestyle=":",  label=f"Spec = {SPEC_MEAN*100:.0f} cm")

        status = "✓ PASS" if ts["pass"] else "✗ FAIL"
        ax.set_title(
            f"Target {i+1}: ({tgt[0]:+.0f}, {tgt[1]:+.0f}, {tgt[2]:.0f}) "
            f"yaw={tgt[3]:.2f} rad — "
            f"mean={mean_cm:.2f} cm  std={ts['std']*100:.2f} cm  {status}",
            fontsize=10,
        )
        ax.set_ylabel("Position error (cm)")
        ax.set_xlim(0, SECS_PER_TARGET)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Simulation time (s)")

    overall_pass = overall_mean < SPEC_MEAN and overall_std < SPEC_STD
    fig.suptitle(
        f"mark.py — Automated Scoring\n"
        f"Overall: mean={overall_mean*100:.2f} cm  std={overall_std*100:.2f} cm  "
        f"{'PASS ✓' if overall_pass else 'FAIL ✗'}  (spec < 1 cm)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("mark_results.png", dpi=150)
    print("Plot saved to mark_results.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Show PyBullet window")
    args = parser.parse_args()
    run_marking(gui=args.gui)
