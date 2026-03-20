"""
GUI demo using the best PID gains found by tune_pid.py.

Run from the assignment_3_2026 directory:
    python demo.py                   # loads best_gains.json
    python demo.py --gains-file best_gains.json
    python demo.py --target 1 1 1.5 0   # override target

Plots position error and yaw error over time after the run.
"""

import argparse
import json
import math
import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

from src.PID_controller import PIDController
from src.tello_controller import TelloController

# ---------------------------------------------------------------------------
# Defaults (match tune_pid.py)
# ---------------------------------------------------------------------------
SIM_HZ   = 1000
CTRL_HZ  = 50
SIM_SECS = 10.0

DRONE_URDF = "resources/tello.urdf"
START_POS  = [0, 0, 1]

M   = 0.088
L   = 0.06
KF  = 0.566e-5
KM  = 0.762e-7
K_TRANS = np.array([3.365e-2, 3.365e-2, 3.365e-2])
TM  = 0.0163


def run_demo(gains, target, realtime=True):
    tgt_x, tgt_y, tgt_z, tgt_yaw = target

    # --- PyBullet GUI setup ---
    client = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)

    start_orn = p.getQuaternionFromEuler([0, 0, 0])
    drone_id  = p.loadURDF(DRONE_URDF, START_POS, start_orn, physicsClientId=client)

    # Red sphere at target
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1],
                              physicsClientId=client)
    p.createMultiBody(0, -1, vis, [tgt_x, tgt_y, tgt_z], physicsClientId=client)

    # --- PID controllers ---
    kp, ki, kd = gains["kp"], gains["ki"], gains["kd"]
    kp_yaw, ki_yaw, kd_yaw = gains["kp_yaw"], gains["ki_yaw"], gains["kd_yaw"]

    pos_pid = PIDController(
        Kp=np.array([kp, kp, kp]),
        Ki=np.array([ki, ki, ki]),
        Kd=np.array([kd, kd, kd]),
        Ki_sat=np.array([1.0, 1.0, 1.0]),
    )
    yaw_pid = PIDController(
        Kp=np.array([kp_yaw, 0.0, 0.0]),
        Ki=np.array([ki_yaw, 0.0, 0.0]),
        Kd=np.array([kd_yaw, 0.0, 0.0]),
        Ki_sat=np.array([1.0, 0.0, 0.0]),
    )
    tello = TelloController(9.81, M, L, 0.35, KF, KM)

    timestep      = 1.0 / SIM_HZ
    ctrl_timestep = 1.0 / CTRL_HZ
    steps_per_ctrl = int(round(ctrl_timestep / timestep))
    total_steps    = int(SIM_SECS * SIM_HZ)

    VEL_LIMIT = 2.0
    YAW_LIMIT = 1.5

    prev_rpm    = np.zeros(4)
    desired_vel = np.zeros(3)
    yaw_rate_sp = 0.0
    loop_counter = 0

    # Logging
    times      = []
    pos_errs   = []
    yaw_errs   = []

    print(f"Running demo  target={target}  gains: pos kp={kp:.4f} ki={ki:.4f} kd={kd:.4f}"
          f"  yaw kp={kp_yaw:.4f} ki={ki_yaw:.4f} kd={kd_yaw:.4f}")
    print("Press Ctrl-C to stop early.")

    try:
        for step in range(total_steps):
            loop_start = time.time()

            pos, quat = p.getBasePositionAndOrientation(drone_id, physicsClientId=client)
            lin_vel_world, ang_vel_world = p.getBaseVelocity(drone_id, physicsClientId=client)

            roll, pitch, yaw = p.getEulerFromQuaternion(quat)
            yaw_quat = p.getQuaternionFromEuler([0, 0, yaw])
            _, inv_quat_yaw = p.invertTransform([0, 0, 0], yaw_quat)
            _, inv_quat     = p.invertTransform([0, 0, 0], quat)

            lin_vel = np.array(p.rotateVector(inv_quat_yaw, lin_vel_world))
            ang_vel = np.array(p.rotateVector(inv_quat,     ang_vel_world))

            loop_counter += 1
            if loop_counter >= steps_per_ctrl:
                loop_counter = 0

                pos_error = np.array([tgt_x - pos[0], tgt_y - pos[1], tgt_z - pos[2]])
                vel_world = pos_pid.control_update(pos_error, ctrl_timestep)
                vel_world = np.clip(vel_world, -VEL_LIMIT, VEL_LIMIT)

                yaw_error  = (tgt_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
                yaw_rate_sp = float(np.clip(
                    yaw_pid.control_update(np.array([yaw_error, 0, 0]), ctrl_timestep)[0],
                    -YAW_LIMIT, YAW_LIMIT,
                ))

                desired_vel = vel_world

                t = step * timestep
                times.append(t)
                pos_errs.append(np.linalg.norm(pos_error))
                yaw_errs.append(abs(yaw_error))

            # Motor dynamics
            rpm = tello.compute_control(desired_vel, lin_vel, quat, ang_vel, yaw_rate_sp, timestep)
            rpm_deriv = (rpm - prev_rpm) / TM
            rpm = prev_rpm + rpm_deriv * timestep
            prev_rpm = rpm

            rotation     = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            omega        = rpm * (2 * np.pi / 60)
            motor_forces = omega**2 * KF
            thrust       = np.array([0, 0, np.sum(motor_forces)])
            vel_body     = rotation.T @ np.array(lin_vel_world)
            force        = -K_TRANS * vel_body + thrust

            z_torques = omega**2 * KM
            torques = np.array([
                (-motor_forces[0] + motor_forces[1] + motor_forces[2] - motor_forces[3]) * L,
                (-motor_forces[0] + motor_forces[1] - motor_forces[2] + motor_forces[3]) * L,
                -z_torques[0] - z_torques[1] + z_torques[2] + z_torques[3],
            ])

            p.applyExternalForce(drone_id, -1, force.tolist(),   [0, 0, 0], p.LINK_FRAME,  physicsClientId=client)
            p.applyExternalTorque(drone_id, -1, torques.tolist(),            p.LINK_FRAME,  physicsClientId=client)
            p.stepSimulation(physicsClientId=client)

            if realtime:
                elapsed = time.time() - loop_start
                if elapsed < timestep:
                    time.sleep(timestep - elapsed)

    except KeyboardInterrupt:
        print("Stopped early.")

    p.disconnect(client)

    # --- Plot ---
    times    = np.array(times)
    pos_errs = np.array(pos_errs)
    yaw_errs = np.array(yaw_errs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(times, pos_errs, color="tab:blue")
    ax1.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Position error (m)")
    ax1.set_title(f"Demo  target={target}  final pos_err={pos_errs[-1]:.4f} m  yaw_err={yaw_errs[-1]:.4f} rad")
    ax1.grid(True)

    ax2.plot(times, yaw_errs, color="tab:orange")
    ax2.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Yaw error (rad)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("demo_result.png", dpi=150)
    print(f"\nPlot saved to demo_result.png")
    print(f"Final position error : {pos_errs[-1]:.4f} m")
    print(f"Final yaw error      : {yaw_errs[-1]:.4f} rad")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gains-file", default="best_gains.json",
                        help="JSON file produced by tune_pid.py (default: best_gains.json)")
    parser.add_argument("--target", nargs=4, type=float, metavar=("X", "Y", "Z", "YAW"),
                        default=None, help="Override target (default: reads from gains file or 1 1 1.5 0)")
    parser.add_argument("--no-realtime", action="store_true",
                        help="Run as fast as possible instead of real-time")
    args = parser.parse_args()

    try:
        with open(args.gains_file) as f:
            gains = json.load(f)
        print(f"Loaded gains from {args.gains_file}  (score={gains.get('score', '?'):.4f})")
    except FileNotFoundError:
        print(f"ERROR: {args.gains_file} not found. Run tune_pid.py first.")
        return

    target = tuple(args.target) if args.target else (1.0, 1.0, 1.5, 0.0)
    run_demo(gains, target, realtime=not args.no_realtime)


if __name__ == "__main__":
    main()
