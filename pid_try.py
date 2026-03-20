wind_flag = True  # Set to True to enable wind testing (+1 mark in marking)

# ============================================================
# Author: Juan Ignacio Doval Roque
#
# Feedback Controller for UAV Position Stabilisation
# Method: PID (Proportional-Integral-Derivative) control
#
# THEORY:
# A PID controller computes a control output based on the error between
# a desired setpoint and the measured state. It has three terms:
#   P (proportional): reacts to current error — drives drone toward target
#   I (integral):     reacts to accumulated error — eliminates steady-state offset
#   D (derivative):   reacts to rate of change of error — damps oscillation
#
# Output = Kp * error + Ki * integral(error) + Kd * d(error)/dt
#
# The controller runs once per timestep (dt). It receives the drone's
# current state, computes position/yaw errors, feeds them through PID,
# then rotates the output from world frame into the drone's body frame
# using a yaw rotation matrix before sending velocity commands.
# ============================================================

import math
import numpy as np

# Import the provided PID controller class
from src.PID_controller import PIDController

# ============================================================
# TUNING GUIDE — READ BEFORE LAB
#
# STEP 1: Start with Kp only (set Ki=Kd=0.0 everywhere).
#         Increase Kp_pos until the drone moves toward target
#         without oscillating wildly. ~0.3-0.8 is a good range.
#
# STEP 2: Add Kd_pos to reduce overshoot/oscillation.
#         Increase until oscillation reduces. ~0.05-0.2.
#
# STEP 3: Add small Ki_pos to fix any remaining steady-state
#         drift (drone settles near but not exactly on target).
#         Keep small — too high causes windup and oscillation. ~0.01-0.05.
#
# STEP 4: Repeat steps 1-3 for yaw (Kp_yaw, Kd_yaw, Ki_yaw).
#         Yaw typically needs higher Kp than position.
#
# STEP 5: Enable wind (press k in sim). Increase Ki if drone
#         drifts under wind — integral action rejects constant disturbances.
#
# STEP 6: Reduce VEL_LIMIT if drone overshoots badly on large moves.
#         Increase if response is too sluggish.
#
# NOTE: Press 'r' in simulator to reload controller after every change.
#       No need to restart the simulation.
# ============================================================

# --- POSITION PID GAINS (x, y, z) ---
Kp_pos  = np.array([0.5,  0.5,  0.5 ])  # TUNE: increase for faster response
Ki_pos  = np.array([0.03, 0.03, 0.03])  # TUNE: increase if drone drifts at setpoint
Kd_pos  = np.array([0.1,  0.1,  0.1 ])  # TUNE: increase to reduce overshoot
Ksat_pos = np.array([1.0,  1.0,  1.0 ])  # integral saturation limit — keep as is

# --- YAW PID GAINS ---
Kp_yaw  = np.array([1.0, 0.0, 0.0])     # TUNE: increase for faster yaw response
Ki_yaw  = np.array([0.02, 0.0, 0.0])    # TUNE: small — yaw rarely needs much I
Kd_yaw  = np.array([0.1, 0.0, 0.0])     # TUNE: increase if yaw oscillates
Ksat_yaw = np.array([1.0, 0.0, 0.0])    # integral saturation limit

# --- OUTPUT LIMITS ---
# Reduce VEL_LIMIT if drone overshoots badly, increase if too sluggish
VEL_LIMIT = 2.0   # max velocity command (m/s) per axis
YAW_LIMIT = 1.5   # max yaw rate command (rad/s)

# Initialise PID controllers at module level so state persists between timesteps.
# These reset automatically when you press 'r' to reload the controller.
pos_pid = PIDController(Kp=Kp_pos, Ki=Ki_pos, Kd=Kd_pos, Ki_sat=Ksat_pos)
yaw_pid = PIDController(Kp=Kp_yaw, Ki=Ki_yaw, Kd=Kd_yaw, Ki_sat=Ksat_yaw)


def controller(state, target_pos, dt, wind_enabled=False):
    # state:      [pos_x, pos_y, pos_z, roll, pitch, yaw]  (metres, radians)
    # target_pos: (x, y, z, yaw)                           (metres, radians)
    # dt:         timestep (seconds)
    # wind_enabled: True if wind disturbance is active
    # return:     (vel_x, vel_y, vel_z, yaw_rate)          (m/s, rad/s)

    # --- EXTRACT CURRENT STATE ---
    cur_x, cur_y, cur_z = state[0], state[1], state[2]
    cur_yaw = state[5]  # roll=state[3], pitch=state[4] not used here

    # --- EXTRACT TARGET ---
    tgt_x, tgt_y, tgt_z, tgt_yaw = target_pos

    # --- POSITION ERROR (world frame) ---
    pos_error = np.array([tgt_x - cur_x,
                          tgt_y - cur_y,
                          tgt_z - cur_z])

    # --- PID: compute world-frame velocity commands ---
    vel_world = pos_pid.control_update(pos_error, dt)

    # --- YAW ERROR ---
    # Wrap to [-pi, pi] so drone always turns the shortest way.
    # Without this, e.g. error of 350deg would turn the long way instead of -10deg.
    yaw_error = tgt_yaw - cur_yaw
    yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

    # yaw_pid expects a 3-element array — only first element used
    yaw_output = yaw_pid.control_update(np.array([yaw_error, 0.0, 0.0]), dt)
    yaw_rate = yaw_output[0]

    # --- AXIS TRANSFORMATION (world frame → body frame) ---
    # The simulator provides state in world frame (global x/y/z) but
    # velocity commands must be in the drone's body frame (rotated by yaw).
    # Apply 2D rotation matrix around z-axis by current yaw angle:
    #   vx_body =  vx_world * cos(yaw) + vy_world * sin(yaw)
    #   vy_body = -vx_world * sin(yaw) + vy_world * cos(yaw)
    # Note: z and yaw are unaffected by this rotation.
    #
    # LAB NOTE: if drone moves in wrong direction when yaw != 0,
    # check this transformation — it's the most common issue on real hardware.
    vx_body =  vel_world[0] * math.cos(cur_yaw) + vel_world[1] * math.sin(cur_yaw)
    vy_body = -vel_world[0] * math.sin(cur_yaw) + vel_world[1] * math.cos(cur_yaw)
    vz_body = vel_world[2]

    # --- CLAMP OUTPUTS ---
    # Prevents extreme commands that could destabilise the drone.
    # If drone is too aggressive reduce VEL_LIMIT at top of file.
    vx_body  = max(-VEL_LIMIT, min(VEL_LIMIT, vx_body))
    vy_body  = max(-VEL_LIMIT, min(VEL_LIMIT, vy_body))
    vz_body  = max(-VEL_LIMIT, min(VEL_LIMIT, vz_body))
    yaw_rate = max(-YAW_LIMIT, min(YAW_LIMIT, yaw_rate))

    return (vx_body, vy_body, vz_body, yaw_rate)
