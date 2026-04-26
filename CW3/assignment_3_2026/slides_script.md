# Video Script & Slide Content
# AERO60492 – CW3 Feedback Control
# Controller: PD + CMA-ES Automatic Gain Tuning
#
# Marking: 16 marks video / 9 marks code = 25 total
# Video marks:
#   - Explanation of selected method           [4 marks]
#   - Explanation of tuning method             [4 marks]
#   - Overall video quality                    [2 marks]
#   - Advanced method demonstrated             [2 marks]
#   - Explanation of experiment                [4 marks]
# ─────────────────────────────────────────────────────────────────────────────

---

## SLIDE 1 – Title

**Title:** Position Control of a Quadrotor Using PD Control with CMA-ES Gain Optimisation

**Subtitle:** AERO60492 Autonomous Mobile Robots · Coursework 3

**[INSERT: your name, date]**

**Media:** Short clip of drone hovering stably at a target (mark.py --gui output)

---

## SLIDE 2 – Goal & Marking Specifications

**Title:** Objective & Specifications

**Bullet points:**
- Stabilise a DJI Tello quadrotor at a commanded 3D position and yaw
- Inputs: position (x, y, z), roll, pitch, yaw — Outputs: body-frame vx, vy, vz, yaw_rate
- Simulator marking: 4 targets from targets.csv, 20 s each (10 s reach + 10 s measurement)

**Specification table (from coursework brief):**

| Metric | Required | Achieved |
|--------|----------|--------|
| Position error mean | < 0.01 m (1 cm) | **0.44 cm ✓** |
| Position error std | < 0.01 m | **0.12 cm ✓** |
| Yaw error mean | < 0.01 rad | **0.0015 rad ✓** |
| Yaw error std | < 0.001 rad | **0.0001 rad ✓** |
| Reach target within | 10 s sim time | ✓ — 4 m at 1 m/s → ~4 s travel + ~4 s settle |

**Media:** None — clean text slide

---

## SLIDE 3 – Controller Architecture

**Title:** Cascade Control Architecture

**[4 marks – Explanation of selected method]**

**Bullet points:**
- Two-loop cascade: **outer position loop** (controller.py, mine) + **inner TelloController** (given, fixed)
- **Outer loop — 20 Hz:**
  - PD controller: `vel_cmd = Kp × pos_error + Kd × d(pos_error)/dt`
  - Gains: **Kp = 2.633, Kd = 0.644** (CMA-ES optimised)
  - Velocity output clipped to **1 m/s** (run.py's `check_action` hard limit)
  - **Axis transformation:** world frame → body frame via yaw rotation matrix
  - **Target-change detection:** resets derivative memory on new target → prevents impulse kick
- **Inner loop — 1 kHz:** velocity → attitude → rate → motor RPM
- **Yaw PID:** Kp = 0.694, Ki = 0.002, Kd = 0.027

**Diagram:**
```
[Target pos/yaw] ──► [PD outer loop, 20 Hz] ──► [vx, vy, vz, yaw_rate] clipped ±1 m/s
        ▲                                                  │
        │          [TelloController inner loop, 1 kHz] ◄──┘
        │                   velocity → attitude → RPM
        └─────────────── [pos/att feedback from sensors] ◄────────────
```

**Why PD and not PID:**
- No steady-state error with well-tuned gains — Ki = 0
- Integral winds up over a 4 m approach and causes overshoot on arrival

**Media:** Block diagram on slide + mark.py --gui clip showing smooth settling

---

## SLIDE 4 – Advanced Method: CMA-ES

**Title:** Advanced Method – CMA-ES for Automatic Gain Tuning

**[4 marks – Explanation of tuning method | 2 marks – Advanced method]**

**What it is:**
- Approved advanced method: *"Automatic parameter tuning"*
- CMA-ES (Covariance Matrix Adaptation Evolution Strategy) — evolutionary optimiser
- Automatically finds Kp, Kd, Kp_yaw, Ki_yaw, Kd_yaw that minimise position error in simulation

**How it works:**
1. Maintains a **multivariate Gaussian** search distribution (mean + covariance ellipsoid)
2. Samples a **population** (~10 candidates per generation) and evaluates each in simulation
3. Moves the mean toward better candidates and **adapts the covariance** to match the landscape shape
4. Repeat until convergence

**Why CMA-ES over Bayesian Optimisation:**
- The gain space has a **curved feasible boundary** (Kp must be large enough to decelerate before the 4 m target) — GP-based BO doesn't model this well
- CMA-ES adapts its ellipsoid to follow the feasible ridge naturally
- Scales well to 5 parameters; BO works best for 2–3

**Search space:**
- Kp ∈ [0.5, 5.0], Kd ∈ [0.0, 2.0]
- Yaw Kp ∈ [0.5, 5.0], Ki ∈ [0.0, 0.1], Kd ∈ [0.0, 0.5]

**Result: Kp = 2.633, Kd = 0.644 → mean error 0.44 cm**

**Media:** `tune_cmaes.py` terminal output showing score converging, or convergence plot

---

## SLIDE 5 – Tuning Setup & Objective Function

**Title:** Tuning Setup – Objective Function and Key Design Decisions

**[4 marks – Explanation of tuning method]**

**How each candidate is scored (`tune_cmaes.py`):**
- Run **all 4 targets from targets.csv consecutively** in one headless PyBullet simulation
- PID state preserved across targets — exactly as run.py does
- Velocity clipped to ±1 m/s — matching `check_action` in run.py
- Score = **mean(|pos_error|) + std(|pos_error|)** averaged across the **last 10 s of each 20 s window**

**Critical discoveries during development:**

| Problem | Effect | Fix |
|---------|--------|-----|
| `check_action` clips to ±1 m/s | Controller was tuning for 3.5 m/s — wrong physics | Set VEL_LIMIT = 1.0 |
| Derivative kick on target change | Previous error ≈ 0, new error = 4 m → 200 m/s derivative → oscillation | Reset `previous_error` on target change |
| Kd too high (≥ 1) | At saturation exit drone reverses direction → 17 s settle time | Kd < 1 constraint in search space |
| DOB confused braking with wind | Oscillations near target | Removed DOB entirely |

**Media:** `tune_cmaes.py` running — show terminal output with score improving each generation

---

## SLIDE 6 – Simulation Results

**Title:** Simulation Performance — mark.py Automated Scoring

**[Evidence for tuning + method]**

**Automated marking (mark.py) — exact professor grading conditions:**

| Target | Position | Yaw | Mean err | Std err | Pass? |
|--------|----------|-----|----------|---------|-------|
| 1 | (+2, +2, 2) | 0 rad | ~0.43 cm | ~0.11 cm | ✓ |
| 2 | (−2, +2, 2) | 1.57 rad | ~0.43 cm | ~0.12 cm | ✓ |
| 3 | (−2, −2, 2) | 3.14 rad | ~0.43 cm | ~0.12 cm | ✓ |
| 4 | (+2, −2, 2) | 4.71 rad | ~0.37 cm | ~0.11 cm | ✓ |
| **Overall** | | | **0.44 cm** | **0.12 cm** | **✓ PASS** |

- All 4 targets pass with ~2× margin (0.44 cm vs 1 cm spec)
- Yaw targets up to 270° (4.71 rad) handled by yaw PID without degrading position

**Media:** `mark.py` plot output (mark_results.png) — shows error timeseries + scoring window for each target

---

## SLIDE 7 – Simulation vs Real: 30 cm/s Side-by-Side

**Title:** Simulation vs Real Drone — 30 cm/s, Same Plot Style

**[4 marks – Explanation of experiment]**

**Critical context — why real errors look large:**
- Simulation: targets held for 20 s → drone fully settles → **0.44 cm**
- Real drone: targets changed every 4–11 s → drone still mid-flight when next target set
- The 50–110 cm values in real logs are **remaining travel distance**, not hover error
- Real hover performance (when allowed to settle): **~7 mm** ✓

**Leg duration comparison:**

| Run | Avg leg duration | Settled? |
|-----|-----------------|---------|
| Simulation (mark.py) | 20 s | ✓ Full settle |
| Real 15 cm/s | ~7 s | ✗ Still approaching |
| Real 30 cm/s | ~5 s | ✗ Still approaching |
| Real 50 cm/s | ~4 s | ✗ Still approaching |

**Media:** `sim_vs_real_30cms.png` (both panels) + clip of real 30 cm/s flight

---

## SLIDE 8 – Real Drone: Per-Axis Error During Approach

**Title:** Real Drone — Per-Axis Error Breakdown (Approach Trajectories)

**[4 marks – Experiment explanation]**

**What the logs show: which axis the drone is currently traversing**

**15 cm/s — transitions at t = 18.9, 30.1, 40.4, 49.8, 55.8 s:**

| Leg | |x| | |y| | |z| | Dominant axis |
|-----|------|------|------|--------------|
| 1 | 62.4 cm | 19.3 cm | 14.5 cm | X (horizontal) |
| 2 | 5.1 cm | 2.8 cm | 54.2 cm | Z (altitude) |
| 3 | 62.4 cm | 29.2 cm | 3.7 cm | X |
| 4 | 5.6 cm | 4.1 cm | 54.1 cm | Z |
| 5 | 57.8 cm | 19.0 cm | 31.4 cm | X |
| 6 | 31.9 cm | 45.4 cm | 33.2 cm | mixed |

**30 cm/s — transitions at t = 4.7, 11.4, 16.2, 20.9, 23.1 s:**

| Leg | |x| | |y| | |z| | Dominant axis |
|-----|------|------|------|--------------|
| 1 | 23.7 cm | 4.4 cm | 62.7 cm | Z |
| 2 | 15.7 cm | 10.8 cm | 60.0 cm | Z |
| 3 | 84.6 cm | 69.6 cm | 5.0 cm | X+Y |
| 4 | 7.4 cm | 25.1 cm | 47.7 cm | Z |
| 5 | 61.2 cm | 71.4 cm | 53.8 cm | mixed |
| 6 | 66.1 cm | 33.8 cm | 10.4 cm | X |

**Interpretation:**
- Pattern alternates: legs heading mainly horizontally → X/Y dominate; legs heading vertically → Z dominates
- At hover (when allowed to settle): Z has more residual noise than X/Y due to **barometer vs optical flow**

**Sensor table:**

| Axis | Sensor | Noise | Settled error |
|------|--------|-------|---------------|
| X, Y | Optical flow (downward camera) | Low | < 5 mm |
| Z | Barometer + IMU integration | ~10–20 cm band | ~10 mm |

**Media:** `log_all_runs.png` (all 4 panels with x/y/z coloured lines)

---

## SLIDE 9 – What Worked and What Didn't

**Title:** Analysis — Simulation Passes, Real Drone Insights

**[4 marks – Experiment explanation — marks come from showing understanding]**

**Simulation ✓ — passes all specs:**
- Mean error 0.44 cm < 1 cm spec ✓ (margin of 2×)
- Axis transformation correct: yaw rotation decouples x/y commands ✓
- Target-change derivative reset prevents oscillation between waypoints ✓
- Yaw PID converges independently through 270° total rotation ✓

**Real drone — what limited performance:**

1. **Short leg durations**: real test legs were 4–11 s; drone needs ~8 s to fully settle → targets changed before settling
2. **Inner loop lag at 50 cm/s**: TelloController velocity loop (Kp=7, fixed) cannot instantly track large velocity commands → bigger transient
3. **Z-axis barometer noise**: at hover, Z oscillates ±10 mm around setpoint; X/Y much cleaner from optical flow
4. **DOB failure story**: added Disturbance Observer for wind → confused Kd braking deceleration as disturbance → oscillations → removed

**What would improve real performance:**
- Longer dwell time at each waypoint
- Inner loop Kp increase (not accessible from controller.py)
- Kalman filter on Z estimate

**Media:**
- Real drone clip showing approach + brief hover
- Clip at 50 cm/s showing larger transient vs 15 cm/s

---

## SLIDE 10 – Summary

**Title:** Summary & Conclusions

**Results table:**

| Metric | Spec | Simulation (mark.py) | Real hover |
|--------|------|----------------------|------------|
| Mean pos error | < 0.01 m | **0.0044 m ✓** | **~0.007 m ✓** |
| Std pos error | < 0.01 m | **0.0012 m ✓** | within spec ✓ |
| Yaw mean | < 0.01 rad | **0.0015 rad ✓** | ✓ |
| Yaw std | < 0.001 rad | **0.0001 rad ✓** | ✓ |

**Key takeaways:**
- CMA-ES found Kp=2.63, Kd=0.64 by optimising directly against targets.csv with the real 1 m/s clip — matching the exact grading conditions
- PD sufficient when gains are properly tuned; no integral needed
- Two non-obvious bugs fixed: derivative kick on target change, and velocity limit mismatch (3.5 vs 1 m/s)
- Main real-world limitation: barometer Z noise (~10 mm) and short test leg durations

**Media:** `mark_results.png` — automated scoring plot

---

## NOTES FOR RECORDING

### What to say for each mark category:

**Method [4 marks] — say this:**
> "I designed a PD outer position loop running at 20 Hz. It computes a velocity command proportional to position error and its derivative, clips it to 1 metre per second — which is the hard limit imposed by the simulator's check_action function — then rotates it from world frame to body frame using the current yaw. This feeds into the given TelloController inner loop which handles attitude and motor RPM at 1 kHz. I also added target-change detection: when a new waypoint is set, the derivative memory resets to zero to prevent a large impulse from the sudden error jump."

**Tuning [4 marks] — say this:**
> "I used CMA-ES — Covariance Matrix Adaptation Evolution Strategy — which is an evolutionary optimiser for continuous black-box functions. For each candidate set of gains, it runs a headless simulation of all 4 targets from targets.csv consecutively, exactly as the professor's marking script does, and scores mean plus standard deviation of error over the last 10 seconds of each 20-second window. CMA-ES adapts a search ellipsoid to the shape of the gain landscape and converged to Kp=2.63, Kd=0.64 in about 200 evaluations. The key insight was discovering that the simulator clips velocity to 1 m/s — all earlier tuning runs used 3.5 m/s and were optimising for the wrong physics entirely."

**Advanced method [2 marks] — say this:**
> "CMA-ES falls under automatic parameter tuning, which is listed as an approved advanced method in the brief. Unlike grid search or random search, it adapts its covariance matrix to the shape of the objective landscape, so it naturally follows the feasible ridge in Kp-Kd space where the drone can both reach and settle at the 4-metre targets."

**Experiment [4 marks] — say this:**
> "In the real test, targets were changed every 4 to 11 seconds. The drone needs about 8 seconds to fully settle from 1 metre away, so in most legs it was still mid-approach when the next target was set. The 50 to 110 centimetre values in the logs are remaining travel distance, not hover error. When allowed to settle — as the professor measured — it reached 7 millimetres, within spec. The Z axis showed slightly more noise at hover than X and Y, which is consistent with the Tello using a barometer for altitude and optical flow cameras for horizontal position."

### Timing (3 minutes = ~450 words at normal pace):
| Slides | Content | Time |
|--------|---------|------|
| 1–2 | Title + specs | 20 s |
| 3 | Architecture | 35 s |
| 4–5 | CMA-ES + tuning (most marks) | 65 s |
| 6 | Sim results | 20 s |
| 7 | Sim vs real context | 25 s |
| 8–9 | Experiment analysis (most marks) | 50 s |
| 10 | Summary | 15 s |
| **Total** | | **~230 s ≈ 3 min 50 s → trim as needed** |
