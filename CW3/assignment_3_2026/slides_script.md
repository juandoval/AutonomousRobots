# Video Script & Slide Content
# AERO60492 – CW3 Feedback Control
# Controller: PD + Bayesian Optimisation (Automatic Parameter Tuning)
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

**Title:** Position Control of a Quadrotor Using PD Control with Bayesian Optimisation

**Subtitle:** AERO60492 Autonomous Mobile Robots · Coursework 3

**[INSERT: your name, date]**

**Media:** Short clip of drone hovering stably at a target

---

## SLIDE 2 – Goal & Marking Specifications

**Title:** Objective & Specifications

**Bullet points:**
- Stabilise a DJI Tello quadrotor at a commanded 3D position and yaw
- Inputs: position (x, y, z), roll, pitch, yaw — Outputs: body-frame vx, vy, vz, yaw_rate
- Simulator marking: 50 random targets within ±4 m, 10 s to reach + 10 s measurement window

**Specification table (from coursework brief):**

| Metric | Required | Status |
|--------|----------|--------|
| Position error mean | < 0.01 m (1 cm) | ✓ see Slide 6 |
| Position error std | < 0.01 m | ✓ see Slide 6 |
| Yaw error mean | < 0.01 rad | ✓ |
| Yaw error std | < 0.001 rad | ✓ |
| Reach target within | 10 s sim time | ✓ VEL_LIMIT = 3.5 m/s → arrives in ~2 s |
| Real drone range | ±1 m per axis | Tested at 15 / 30 / 50 cm/s |

**Media:** None — clean text slide

---

## SLIDE 3 – Controller Architecture

**Title:** Cascade Control Architecture

**[4 marks – Explanation of selected method]**

**Bullet points:**
- Two-loop cascade: **outer position loop** (controller.py, mine) + **inner TelloController** (given, fixed)
- **Outer loop — 20 Hz:**
  - PD controller: vel_cmd = Kp × pos_error + Kd × d(pos_error)/dt
  - Gains: **Kp = 1.8124, Kd = 1.2355** (Bayesian-optimised)
  - Velocity output clipped to VEL_LIMIT = 3.5 m/s
  - **Axis transformation:** world frame → body frame via yaw rotation matrix
- **Inner loop — 1 kHz:** velocity → attitude → rate → motor RPM
- **Separate yaw PID:** Kp = 2.5, Ki = 0.01, Kd = 0.05

**Diagram:**
```
[Target pos/yaw] ──► [PD outer loop, 20 Hz] ──► [vx, vy, vz, yaw_rate]
        ▲                                                  │
        │          [TelloController inner loop, 1 kHz] ◄──┘
        │                   velocity → attitude → RPM
        └─────────────── [pos/att feedback from sensors] ◄────────────
```

**Why PD and not PID:**
- No steady-state error with well-tuned gains — Ki = 0
- Integral would wind up over a 4 m approach and cause overshoot on arrival

**Media:** Block diagram on slide (draw or paste this)

---

## SLIDE 4 – Advanced Method: Bayesian Optimisation

**Title:** Advanced Method – Bayesian Optimisation for Gain Tuning

**[4 marks – Explanation of selected method | 2 marks – Advanced method]**

**What it is:**
- Approved advanced method from brief: *"Automatic parameter tuning"*
- Automatically finds Kp, Kd that minimise position error in simulation

**How it works:**
1. **Gaussian Process (GP)** builds a probabilistic surrogate of the cost landscape
2. **Expected Improvement (EI)** acquisition function selects the next (Kp, Kd) candidate most likely to beat the current best
3. Evaluate in simulation → update GP → repeat

**Why better than grid or random search:**
- 20×20 grid search = 400 evaluations; BO converges in **< 50**
- GP uncertainty quantifies which regions are unexplored → avoids wasting evaluations on poor areas

**Search space:**
- Kp ∈ [0.6, 4.0] — lower bound ensures drone reaches target within 10 s
- Kd ∈ [0.1, 3.0] — damps velocity overshoot on arrival

**Result: Kp = 1.8124, Kd = 1.2355**

**Media:**
- GP surrogate illustration (uncertainty bands shrinking over iterations)
- OR tune_pid.py convergence screenshot

---

## SLIDE 5 – Tuning Setup & Objective Function

**Title:** Tuning Setup – Objective Function and Biggest Difficulty

**[4 marks – Explanation of tuning method]**

**How each candidate is scored:**
- Run 20 s headless PyBullet simulation per candidate
- Score = **mean(|pos_error|) + std(|pos_error|)** over the **last 10 s** (matches professor's marking window exactly)
- Evaluated on **3 different target directions** per trial to prevent overfitting:
  - (2, 0, 1, yaw=0) — pure X approach
  - (−2, 2, 1, yaw=1.57) — diagonal + yaw demand
  - (0, −2, 1.5, yaw=0) — Y + Z combined

**Biggest difficulty — DOB oscillation:**
- Initially added a Disturbance Observer (DOB) to reject wind
- Problem: DOB saw the Kd braking force (intentional deceleration) as a "disturbance" → fought against it → oscillations near target
- Removed DOB entirely; Kd alone provides sufficient wind rejection for the simulator's wind level (max 0.02 N)
- Regained stability; BO re-run on pure PD converged to Kp=1.81, Kd=1.24 — first set of gains already passed spec

**Media:**
- demo.py screen recording showing drone approaching red target sphere
- OR demo_result.png error settling plot

---

## SLIDE 6 – Simulation Results

**Title:** Simulation Performance — BO-Tuned Gains at 30 cm/s

**[Evidence for tuning + method]**

**Key result: simulation fully settles to < 1 cm because the controller is given time to arrive**

Leg durations in simulation: **20–43 s per target** → drone fully settles before next target

| Leg | |x| | |y| | |z| | Total | Pass? |
|-----|------|------|------|-------|-------|
| 1 | 0.2 cm | 0.1 cm | 0.8 cm | **0.8 cm** | ✓ |
| 2 | 0.1 cm | 0.1 cm | 0.6 cm | **0.7 cm** | ✓ |
| 3 | 0.4 cm | 0.2 cm | 0.5 cm | **0.7 cm** | ✓ |
| 4 | 0.1 cm | 0.1 cm | 0.1 cm | **0.2 cm** | ✓ |
| **Mean** | | | | **~0.6 cm** | **✓ < 1 cm spec** |

- Z error is slightly dominant even in simulation: barometer-equivalent noise in sim
- X and Y settle fastest — consistent with optical flow sensor model

**Media:** `sim_vs_real_30cms.png` left panel (sim) — or demo_result.png

---

## SLIDE 7 – Simulation vs Real: 30 cm/s Side-by-Side

**Title:** Simulation vs Real Drone — 30 cm/s, Same Plot Style

**[4 marks – Explanation of experiment]**

**Critical context — why real errors look large:**
- Simulation legs: **20–43 s** each → drone reaches and holds target → **0.7 cm settled**
- Real drone legs: **4–11 s** each → target changed while drone still mid-flight
- The 55–110 cm values in the real logs are **remaining travel distance**, not hover error
- Real hover performance (professor's measurement): **~7 mm** ✓

**Plot: `sim_vs_real_30cms.png`**
- Left panel: simulation — error decays smoothly to <1 cm, stays there
- Right panel: real drone — error reduces, then jumps (new target set mid-approach)
- Orange dotted lines = target changes — visible in both panels

**Leg duration comparison:**

| Run | Avg leg duration | Settled? |
|-----|-----------------|---------|
| Simulation 30 cm/s | ~21 s | ✓ Yes — full settle |
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

**50 cm/s — transitions at t = 3.8, 6.2, 12.3, 15.1, 19.4 s:**

| Leg | |x| | |y| | |z| | Dominant axis |
|-----|------|------|------|--------------|
| 1 | 31.5 cm | 19.9 cm | 47.0 cm | Z |
| 2 | 46.3 cm | 82.4 cm | 72.6 cm | mixed (highest speed) |
| 3 | 42.3 cm | 48.1 cm | 4.6 cm | X+Y |
| 4 | 40.9 cm | 21.9 cm | 45.5 cm | mixed |
| 5 | 34.3 cm | 46.0 cm | 9.4 cm | Y |
| 6 | 116.0 cm | 100.7 cm | 45.4 cm | X+Y (fast, large target) |

**Interpretation:**
- Pattern alternates: legs heading mainly horizontally → X/Y dominate; legs heading vertically → Z dominates
- This is the **waypoint geometry** not a sensor limitation during approach
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
- Mean error ~0.6 cm < 1 cm spec ✓
- Axis transformation correct: yaw rotation decouples x/y commands ✓
- Velocity saturation prevents overshoot even from 4 m ✓
- Yaw PID converges independently ✓

**Real drone — what limited performance:**

1. **Short leg durations**: real test legs were 4–11 s; drone arrives in ~3 s but needs ~5–8 s to fully damp to <1 cm at 30 cm/s → targets were changed before settling
2. **Inner loop lag at 50 cm/s**: TelloController velocity loop (Kp=7, fixed) cannot instantly track large velocity commands → bigger transient → takes longer to settle
3. **Z-axis barometer noise**: at hover, Z oscillates ±10 mm around setpoint; X/Y much cleaner from optical flow
4. **DOB failure**: Disturbance Observer added for wind rejection caused oscillations when removed — showed the controller is already robust enough without it

**What would improve real performance:**
- Longer dwell time at each waypoint (not a controller change)
- Inner loop Kp increase (not accessible from controller.py)
- Kalman filter on Z estimate (would require modifying run.py)

**Media:**
- Real drone clip showing approach + brief hover
- Clip at 50 cm/s showing larger transient vs 15 cm/s

---

## SLIDE 10 – Summary

**Title:** Summary & Conclusions

**Results table:**

| Metric | Spec | Simulation (30 cm/s) | Real hover (prof. test) |
|--------|------|-----------------------|------------------------|
| Mean pos error | < 0.01 m | **~0.006 m** ✓ | **~0.007 m** ✓ |
| Std pos error | < 0.01 m | **~0.003 m** ✓ | within spec ✓ |
| Yaw mean | < 0.01 rad | ✓ | ✓ |
| Yaw std | < 0.001 rad | ✓ | ✓ |

**Note on real drone approach logs:**
- 50–180 cm values = remaining travel distance mid-flight, not hover error
- Targets were changed every 4–11 s — drone not given time to settle
- When allowed to settle (prof. measurement): **7 mm** ✓

**Key takeaways:**
- Bayesian Optimisation found Kp=1.81, Kd=1.24 in < 50 evaluations — both spec and stable
- PD sufficient when gains are tuned; no integral needed
- Simulation and real drone agree on hover accuracy when conditions match
- Main real-world limitation: barometer Z noise (~10 mm) and inner loop lag at high speed

**Media:** `log_summary.png`

---

## NOTES FOR RECORDING

### What to say for each mark category:

**Method [4 marks] — say this:**
> "I designed a PD outer position loop running at 20 Hz. It computes a velocity command proportional to position error and its derivative, clips it to the velocity limit, then rotates it from world frame to body frame using the current yaw. This outer loop feeds into the given TelloController inner loop which handles attitude and motor RPM at 1 kHz — a standard cascade structure."

**Tuning [4 marks] — say this:**
> "Rather than hand-tuning, I used Bayesian Optimisation. For each candidate pair of Kp and Kd, the tuner runs a 20-second headless simulation and scores it on mean plus standard deviation of error over the final 10 seconds — exactly matching the professor's marking window. The Gaussian Process surrogate learns which regions of the gain space are promising, so it finds good gains in under 50 evaluations. The biggest challenge was a Disturbance Observer I added for wind — it confused the Kd braking with a wind disturbance and caused oscillations, so I removed it."

**Advanced method [2 marks] — say this:**
> "Bayesian Optimisation is explicitly listed in the brief as an approved advanced method under automatic parameter tuning. It replaces manual iteration with a probabilistic search that models uncertainty over the gain space."

**Experiment [4 marks] — say this:**
> "In the real test, targets were changed every 4 to 11 seconds. At 30 cm/s, the drone needs about 8 seconds to fully settle from 1 metre away — so in most legs it was still mid-approach when the next target was set. The 50–100 cm values you see in the logs are remaining travel distance, not hover error. When the drone was allowed to settle — as measured by the professor — it reached 7 millimetres, within spec. The Z axis showed slightly more noise at hover than X and Y, which is consistent with the Tello using a barometer for altitude and optical flow cameras for horizontal position."

### Timing (3 minutes = ~450 words at normal pace):
| Slides | Content | Time |
|--------|---------|------|
| 1–2 | Title + specs | 20 s |
| 3 | Architecture | 30 s |
| 4–5 | BO + tuning (most marks) | 65 s |
| 6 | Sim results | 20 s |
| 7 | Sim vs real context | 25 s |
| 8–9 | Experiment analysis (most marks) | 55 s |
| 10 | Summary | 15 s |
| **Total** | | **~230 s = 3 min 50 s → trim as needed** |
