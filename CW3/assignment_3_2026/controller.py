wind_flag = True
# Implement a controller

from src.PID_controller import PIDController

import math
import numpy as np

# Position and Yaw PID gains from BO
Kp_pos   = np.array([1.8124, 1.8124, 1.8124])
Ki_pos   = np.array([0.0000, 0.0000, 0.0000])
Kd_pos   = np.array([1.2355, 1.2355, 1.2355])
Kp_yaw   = np.array([0.1000, 0.0, 0.0])
Ki_yaw   = np.array([0.0000, 0.0, 0.0])
Kd_yaw   = np.array([0.0428, 0.0, 0.0])

# Integral saturation limits
Ksat_yaw = np.array([1.0, 0.0, 0.0])    # integral saturation limit
Ksat_pos = np.array([1.0, 1.0, 1.0])  # integral saturation limit

# Output limits
VEL_LIMIT = 3.5   # max velocity command (m/s) per axis — needs to be high enough for 3m targets in 10s
YAW_LIMIT = 1.5   # max yaw rate command (rad/s)

# Iniialise PID controllers
pos_pid = PIDController(Kp=Kp_pos, Ki=Ki_pos, Kd=Kd_pos, Ki_sat=Ksat_pos)
yaw_pid = PIDController(Kp=Kp_yaw, Ki=Ki_yaw, Kd=Kd_yaw, Ki_sat=Ksat_yaw)

def controller(state, target_pos, dt, wind_enabled=False):
    # state:      [pos_x, pos_y, pos_z, roll, pitch, yaw]  (metres, radians)
    # target_pos: (x, y, z, yaw)                           (metres, radians)
    # dt:         timestep (seconds)
    # wind_enabled: True if wind disturbance is active
    # return:     (vel_x, vel_y, vel_z, yaw_rate)          (m/s, rad/s)
    
    # EXTRACT CURRENT STATE
    cur_x, cur_y, cur_z = state[0], state[1], state[2]
    cur_yaw = state[5]  # roll=state[3], pitch=state[4] just for reference
    
    # EXTRACT TARGET
    tgt_x, tgt_y, tgt_z, tgt_yaw = target_pos
    
    # POSITION ERROR (global frame)
    pos_error = np.array([tgt_x - cur_x,
                          tgt_y - cur_y,
                          tgt_z - cur_z])
    
    # PID: compute global-frame velocity commands
    vel_world = pos_pid.control_update(pos_error, dt)
    
    # YAW ERROR
    yaw_error = tgt_yaw - cur_yaw
    yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]
    
    # PID: compute yaw rate command
    yaw_rate = yaw_pid.control_update(np.array([yaw_error, 0, 0]), dt)[0]
    
    # AXIS TRANSFORMATION (global frame → body frame)
    vx_body = vel_world[0] * math.cos(cur_yaw) + vel_world[1] * math.sin(cur_yaw)
    vy_body = -vel_world[0] * math.sin(cur_yaw) + vel_world[1] * math.cos(cur_yaw)
    vz_body = vel_world[2]
    
    # CLAMP OUTPUTS TO LIMITS
    vx_body = np.clip(vx_body, -VEL_LIMIT, VEL_LIMIT)
    vy_body = np.clip(vy_body, -VEL_LIMIT, VEL_LIMIT)
    vz_body = np.clip(vz_body, -VEL_LIMIT, VEL_LIMIT)
    yaw_rate = np.clip(yaw_rate, -YAW_LIMIT, YAW_LIMIT)
    
    output = (vx_body, vy_body, vz_body, yaw_rate)
    return output