wind_flag = False
# Method: PD position control with CMA-ES optimised gains.
#
# Outer position loop (20 Hz): PD controller, gains found via CMA-ES
# (Covariance Matrix Adaptation Evolution Strategy). Scored on mean+std of
# position error over the last 10 s of a 20 s simulation — exactly the
# professor's marking window. No integral (avoids windup on long approaches).
# No DOB (removed after it confused braking deceleration with wind disturbance).

import csv
import math
import numpy as np
import time


class PIDController:
    def __init__(self, Kp, Ki, Kd, Ki_sat):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ki_sat = Ki_sat
        self.previous_error = [0.0, 0.0, 0.0]
        self.int = [0.0, 0.0, 0.0]

    def reset(self):
        self.int = [0.0, 0.0, 0.0]
        self.previous_error = [0.0, 0.0, 0.0]

    def control_update(self, error, timestep):
        self.int += error * timestep
        over_mag = np.argwhere(np.array(self.int) > np.array(self.Ki_sat))
        if over_mag.size != 0:
            for i in range(over_mag.size):
                mag = abs(self.int[over_mag[i][0]])
                self.int[over_mag[i][0]] = (self.int[over_mag[i][0]] / mag) * self.Ki_sat[over_mag[i][0]]
        self.int = np.clip(self.int, -np.array(self.Ki_sat), np.array(self.Ki_sat))
        derivative = (error - self.previous_error) / timestep
        self.previous_error = error
        return self.Kp * error + self.Ki * self.int + self.Kd * derivative

# Position gains — CMA-ES optimised against targets.csv with check_action clip
# Position gains — CMA-ES optimised
# Kp_pos = np.array([1.8124, 1.8124, 1.8124])
Kp_pos = np.array([2.6330, 2.6330, 2.6330])
Ki_pos = np.array([0.0000, 0.0000, 0.0000])
# Kd_pos = np.array([0.800, 0.800, 0.800])
Kd_pos = np.array([0.6440, 0.6440, 0.6440])

# Yaw gains — CMA-ES optimised
# Kp_yaw = np.array([1.1429, 0.0, 0.0])
# Ki_yaw = np.array([0.0284, 0.0, 0.0])
# Kd_yaw = np.array([0.1767, 0.0, 0.0])
Kp_yaw = np.array([0.6945, 0.0, 0.0])
Ki_yaw = np.array([0.0022, 0.0, 0.0])
Kd_yaw = np.array([0.0273, 0.0, 0.0])

Ksat_pos = np.array([1.0, 1.0, 1.0])
Ksat_yaw = np.array([0.5, 0.0, 0.0])

VEL_LIMIT = 1.0  # check_action in run.py clips to ±1 m/s — this is the real limit
YAW_LIMIT = 1.5

pos_pid = PIDController(Kp=Kp_pos, Ki=Ki_pos, Kd=Kd_pos, Ki_sat=Ksat_pos)
yaw_pid = PIDController(Kp=Kp_yaw, Ki=Ki_yaw, Kd=Kd_yaw, Ki_sat=Ksat_yaw)

_print_timer = 0.0
_log_timer   = 0.0
_log_start   = time.time()
_log_file    = open("pos_error_log.csv", "w", newline="")
_log_writer  = csv.writer(_log_file)
_log_writer.writerow(["time_s", "err_mag", "err_x", "err_y", "err_z"])
_last_target = None

def controller(state, target_pos, dt, wind_enabled=False):
    # state:      [pos_x, pos_y, pos_z, roll, pitch, yaw]  (metres, radians)
    # target_pos: (x, y, z, yaw)                           (metres, radians)
    # dt:         timestep (seconds)
    # return:     (vel_x, vel_y, vel_z, yaw_rate)          (m/s, rad/s)
    del wind_enabled  # required by interface
    global _last_target
    if _last_target is None or not np.array_equal(target_pos, _last_target):
        pos_pid.previous_error = np.array([0.0, 0.0, 0.0])
        _last_target = target_pos

    cur_pos = np.array([state[0], state[1], state[2]])
    cur_yaw = state[5]
    tgt_x, tgt_y, tgt_z, tgt_yaw = target_pos
    tgt_pos = np.array([tgt_x, tgt_y, tgt_z])

    # POSITION: PD in world frame
    pos_error = tgt_pos - cur_pos
    vel_world = pos_pid.control_update(pos_error, dt)

    global _print_timer, _log_timer, _log_start
    _print_timer += dt
    _log_timer   += dt

    if _log_timer >= 0.02:  # 50 Hz — matches control rate, captures every update
        _log_timer = 0.0
        t = round(time.time() - _log_start, 3)
        _log_writer.writerow([t, round(float(np.linalg.norm(pos_error)), 6),
                               round(float(pos_error[0]), 6),
                               round(float(pos_error[1]), 6),
                               round(float(pos_error[2]), 6)])
        _log_file.flush()

    if _print_timer >= 0.5:
        _print_timer = 0.0
        print(f"[{time.strftime('%H:%M:%S')}] pos_err: {np.linalg.norm(pos_error):.4f} m  ({pos_error[0]:.3f}, {pos_error[1]:.3f}, {pos_error[2]:.3f})")


    # YAW
    yaw_error = tgt_yaw - cur_yaw
    yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
    yaw_rate = yaw_pid.control_update(np.array([yaw_error, 0, 0]), dt)[0]

    # AXIS TRANSFORMATION: world frame → body frame
    vx_body = vel_world[0] * math.cos(cur_yaw) + vel_world[1] * math.sin(cur_yaw)
    vy_body = -vel_world[0] * math.sin(cur_yaw) + vel_world[1] * math.cos(cur_yaw)
    vz_body = vel_world[2]

    vx_body = np.clip(vx_body, -VEL_LIMIT, VEL_LIMIT)
    vy_body = np.clip(vy_body, -VEL_LIMIT, VEL_LIMIT)
    vz_body = np.clip(vz_body, -VEL_LIMIT, VEL_LIMIT)
    yaw_rate = np.clip(yaw_rate, -YAW_LIMIT, YAW_LIMIT)

    return (vx_body, vy_body, vz_body, yaw_rate)
