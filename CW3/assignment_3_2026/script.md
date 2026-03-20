Video script:

Controller.py (~45 sec)

"The controller implements a cascaded PD architecture. The outer loop computes position error in world frame and passes it through a PID controller to produce a desired velocity. That velocity goes into the TelloController which handles the inner attitude loop and outputs motor RPMs. A separate yaw PID runs in parallel producing a yaw rate setpoint. Key finding from tuning: the integral term was removed entirely — pure PD is optimal for a 10-second stabilisation task."

tune_pid.py (~75 sec)

"Rather than manual tuning, I used Bayesian Optimisation via scikit-optimize. Each trial runs a full headless 10-second PyBullet simulation — matching the professor's test window exactly — and scores the mean position plus weighted yaw error over the last 4 steady-state seconds. A Gaussian Process surrogate learns the gain landscape and uses Expected Improvement to pick the next candidate, making it far more sample-efficient than grid search. I ran two phases: first a broad 200-trial search which revealed the initial bounds were too narrow — kp and kd were hitting the ceiling. A second 150-trial run with wider bounds found the true optimum at kp=2.19, kd=0.95, achieving 0.0039 m steady-state error."

Important notes:

No wind in the optimisation — gains may degrade slightly under wind disturbance
Ki=0 emerged naturally across 350+ trials — don't add it back
Yaw and position are coupled — high kp_yaw degrades position accuracy
Gains are valid for the specific target geometry (1, 1, 1.5) — a very different target may need re-tuning

Why it works in sim
The sim is a perfect model — the physics, motor response, drag coefficients are all exactly what the controller expects. The PID has no surprises.

Why it might not work on a real drone

Model mismatch — KF, KM, K_TRANS are estimated constants. On the real Tello they'll be slightly off, meaning the thrust you command isn't the thrust you get. Your kp=2.19 was tuned assuming those exact constants.

Sensor noise — position in sim is perfect. On a real drone you get noisy GPS/optical flow. The derivative term kd=0.95 is particularly sensitive to this — it amplifies noise because it differentiates the error signal. High kd that works perfectly in sim can cause chatter/oscillation on real hardware.

Latency — the sim runs the control loop at exactly 50 Hz with zero delay. Real hardware has communication delay, sensor delay, actuator lag. Delay destabilises derivative terms especially.

Ki=0 might hurt on real hardware — in sim there's no steady wind, no motor imbalance, no battery sag. On a real drone these create constant disturbances that only the integral term can cancel. Your BO correctly removed Ki because the sim has none of these, but a real drone may drift without it.

What to say to the professor

"The gains were optimised against a high-fidelity headless sim. The high kd was tuned assuming perfect position feedback — on real hardware with noisy sensors this could cause derivative kick. Additionally, Ki=0 emerged from the optimisation because the sim has no persistent disturbances, but a real drone would need some integral action to reject motor imbalance and battery effects. The sim-to-real gap means these gains are a starting point, not a final answer."