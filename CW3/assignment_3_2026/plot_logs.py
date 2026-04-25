"""
plot_logs.py — real drone flight log plots + sim vs real comparison

Outputs:
  log_all_runs.png        — all real runs, per-axis (x/y/z) error
  sim_vs_real_30cms.png   — sim vs real side-by-side, same style (needs log/pos_error_sim_30.csv)
  comparison_30cms.png    — approach-aligned overlay (sim red / real blue)
  log_summary.png         — mean/std bar chart vs spec

After running simulation at VEL_LIMIT=0.30:
  copy pos_error_log.csv → log/pos_error_sim_30.csv
  then re-run this script
"""

import csv, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LOG_DIR = os.path.join(os.path.dirname(__file__), "log")
SPEC    = 0.01  # metres


def load(fname):
    path = os.path.join(LOG_DIR, fname)
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows])
            for k in ("time_s", "err_mag", "err_x", "err_y", "err_z")}


def find_transitions(mag, thr=0.5):
    return list(np.where(np.diff(mag) > thr)[0] + 1)


def draw_run_panel(ax, d, label):
    """Draw one per-axis error panel — same style used for every run."""
    t = d["time_s"] - d["time_s"][0]
    ax.plot(t, d["err_mag"],        "k-",  lw=1.2, label="|total|", zorder=4)
    ax.plot(t, np.abs(d["err_x"]), "r--", lw=0.9, alpha=0.8, label="|x|")
    ax.plot(t, np.abs(d["err_y"]), "g--", lw=0.9, alpha=0.8, label="|y|")
    ax.plot(t, np.abs(d["err_z"]), "b--", lw=0.9, alpha=0.8, label="|z|")
    for idx in find_transitions(d["err_mag"]):
        ax.axvline(t[idx], color="orange", lw=1.0, ls=":", alpha=0.7,
                   label="new target" if idx == find_transitions(d["err_mag"])[0] else "_")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)


def approach_segments(d):
    trans = find_transitions(d["err_mag"])
    bounds = list(zip([0] + trans, trans + [len(d["time_s"])]))
    segs = []
    for s, e in bounds:
        if e - s < 3:
            continue
        t0 = d["time_s"][s]
        segs.append({k: d[k][s:e] for k in d})
        segs[-1]["time_s"] = segs[-1]["time_s"] - t0  # relative
    return segs


# ── 1. All real runs: per-axis ───────────────────────────────────────────────
REAL_LOGS = [
    ("15 cm/s",    "pos_error_log_15.csv"),
    ("30 cm/s",    "pos_error_log30.csv"),
    ("50 cm/s",    "pos_error_log50.csv"),
    ("50 cm/s #2", "pos_error_log_50_2.csv"),
]

fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9))
fig1.suptitle("Real Drone — Position Error (|x|, |y|, |z|, magnitude)", fontsize=12)

for ax, (label, fname) in zip(axes1.flat, REAL_LOGS):
    d = load(fname)
    if d is None:
        ax.set_title(f"{label} — file missing"); continue
    draw_run_panel(ax, d, label)

plt.tight_layout()
plt.savefig("log_all_runs.png", dpi=150)
print("Saved: log_all_runs.png")


# ── 2. Sim vs Real — side-by-side, identical style ──────────────────────────
d_real = load("pos_error_log30.csv")
d_sim  = load("pos_error_sim_30.csv")

fig_sv, axes_sv = plt.subplots(1, 2, figsize=(13, 4), sharey=False)
fig_sv.suptitle("Simulation vs Real Drone — 30 cm/s  (same plot style)", fontsize=12)

if d_sim:
    draw_run_panel(axes_sv[0], d_sim, "Simulation — 30 cm/s")
else:
    axes_sv[0].set_title("Simulation — run sim first\n(copy pos_error_log.csv → log/pos_error_sim_30.csv)")
    axes_sv[0].text(0.5, 0.5, "NO DATA YET", transform=axes_sv[0].transAxes,
                    ha="center", va="center", fontsize=14, color="grey")

if d_real:
    draw_run_panel(axes_sv[1], d_real, "Real Drone — 30 cm/s")

plt.tight_layout()
plt.savefig("sim_vs_real_30cms.png", dpi=150)
print("Saved: sim_vs_real_30cms.png")


# ── 3. Sim — marking window annotated (uses VEL_LIMIT=3.5 run) ──────────────
# Spec: drone has 10 s to reach target, then 10 s of measurement (t=10→20 s)
REACH_T  = 10.0   # seconds allowed to reach target
MEAS_T   = 20.0   # measurement ends at this time after target set
SPEC_ERR = 0.01   # 1 cm spec

# Use the 3.5 m/s run for the spec window (matches actual marking conditions).
# Run:  python run.py  (VEL_LIMIT=3.5, default), let each target settle ~20 s,
#       then copy pos_error_log.csv → log/pos_error_sim_35_2.csv
d_sim_spec = load("pos_error_sim_35_2.csv") or d_sim  # fall back to 30cm/s if not yet recorded

if d_sim_spec:
    src_label = "3.5 m/s" if load("pos_error_sim_35_2.csv") else "30 cm/s (fallback — run 3.5 m/s sim)"
    segs = approach_segments(d_sim_spec)
    n = len(segs)
    fig_spec, axes_spec = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes_spec = [axes_spec]
    fig_spec.suptitle(
        f"Simulation ({src_label}) — instructor measurement window (t = 10–20 s per target)",
        fontsize=11
    )

    all_window_means = []
    all_settled_means = []
    for i, (ax_s, seg) in enumerate(zip(axes_spec, segs)):
        t_rel = seg["time_s"]
        t_end = t_rel[-1]

        # plot all axes
        ax_s.plot(t_rel, seg["err_mag"],        "k-",  lw=1.3, label="|total|", zorder=4)
        ax_s.plot(t_rel, np.abs(seg["err_x"]), "r--", lw=0.8, alpha=0.8, label="|x|")
        ax_s.plot(t_rel, np.abs(seg["err_y"]), "g--", lw=0.8, alpha=0.8, label="|y|")
        ax_s.plot(t_rel, np.abs(seg["err_z"]), "b--", lw=0.8, alpha=0.8, label="|z|")

        # shade the full measurement window (t=10 → 20)
        win_end = min(MEAS_T, t_end)
        ax_s.axvspan(REACH_T, win_end, color="limegreen", alpha=0.12,
                     label="measurement window (t=10–20 s)")
        ax_s.axvline(REACH_T, color="green", lw=1.2, ls="--",
                     label="t=10 s (measurement starts)")

        # spec line
        ax_s.axhline(SPEC_ERR, color="red", lw=1.0, ls=":",
                     label=f"spec = {SPEC_ERR*100:.0f} cm")

        # ── mean over full window (t=10→20) — what instructor measures ──
        mask_win = (t_rel >= REACH_T) & (t_rel <= MEAS_T)
        if mask_win.sum() > 0:
            mean_win = float(seg["err_mag"][mask_win].mean())
            all_window_means.append(mean_win)
            ax_s.axhline(mean_win, color="darkgreen", lw=1.5, ls="-.",
                         label=f"window mean = {mean_win*100:.2f} cm")
            col = "darkgreen" if mean_win < SPEC_ERR else "red"
            verdict = "✓" if mean_win < SPEC_ERR else "✗"
            ax_s.text(REACH_T + 0.3, mean_win * 1.05,
                      f"{verdict} window mean = {mean_win*100:.2f} cm",
                      color=col, fontsize=7.5, va="bottom", fontweight="bold")

        # ── mean over last 5 s of window (fully settled phase) ──
        settle_start = max(REACH_T, MEAS_T - 5.0)
        mask_set = (t_rel >= settle_start) & (t_rel <= MEAS_T)
        if mask_set.sum() > 1:
            mean_set = float(seg["err_mag"][mask_set].mean())
            all_settled_means.append(mean_set)
            ax_s.axhline(mean_set, color="steelblue", lw=1.2, ls="--",
                         label=f"settled mean = {mean_set*100:.2f} cm")
            ax_s.text(win_end - 0.3, mean_set * 0.6,
                      f"settled: {mean_set*100:.2f} cm",
                      color="steelblue", fontsize=7, ha="right", va="top")

        ax_s.set_title(f"Leg {i+1}", fontsize=10)
        ax_s.set_xlabel("Time since target set (s)")
        ax_s.set_ylabel("Error (m)" if i == 0 else "")
        ax_s.set_xlim(left=0)
        ax_s.set_ylim(bottom=0)
        ax_s.legend(fontsize=6, loc="upper right")
        ax_s.grid(True, alpha=0.3)

    if all_window_means:
        ow = float(np.mean(all_window_means))
        os_ = float(np.mean(all_settled_means)) if all_settled_means else float("nan")
        fig_spec.text(
            0.5, 0.01,
            f"MARKING RESULT — window mean (t=10–20 s): {ow*100:.2f} cm  "
            f"{'✓ PASS' if ow < SPEC_ERR else '✗ FAIL'}    |    "
            f"Settled mean (last 5 s): {os_*100:.2f} cm  "
            f"{'✓' if os_ < SPEC_ERR else '✗'}    "
            f"[spec < 1.00 cm]",
            ha="center", fontsize=9,
            color="darkgreen" if ow < SPEC_ERR else "red",
            fontweight="bold"
        )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("sim_spec_window.png", dpi=150)
    print("Saved: sim_spec_window.png")
else:
    print("sim_spec_window.png: no sim data yet — run python run.py, then copy pos_error_log.csv → log/pos_error_sim_35_2.csv")


# ── 4. Approach-aligned overlay (sim vs real) ────────────────────────────────
d_sim  = load("pos_error_sim_30.csv")

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle("Simulation vs Real — 30 cm/s approaches aligned", fontsize=12)

# left: full run overlay
ax = axes2[0]
if d_real:
    t = d_real["time_s"] - d_real["time_s"][0]
    ax.plot(t, d_real["err_mag"], "b-", lw=1.0, label="Real")
    for idx in find_transitions(d_real["err_mag"]):
        ax.axvline(t[idx], color="blue", lw=0.7, ls=":", alpha=0.5)
if d_sim:
    t = d_sim["time_s"] - d_sim["time_s"][0]
    ax.plot(t, d_sim["err_mag"], "r--", lw=1.0, label="Simulation")
    for idx in find_transitions(d_sim["err_mag"]):
        ax.axvline(t[idx], color="red", lw=0.7, ls=":", alpha=0.5)
ax.set_title("Full run (dotted = new target)")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Error (m)")
ax.set_ylim(bottom=0); ax.legend(); ax.grid(True, alpha=0.3)

# middle: approaches aligned by target-set time
ax2 = axes2[1]
for d, color, label in [(d_real, "blue", "Real"), (d_sim, "red", "Sim")]:
    if d is None:
        continue
    for i, seg in enumerate(approach_segments(d)):
        ax2.plot(seg["time_s"], seg["err_mag"], color=color, lw=1.0,
                 alpha=0.5, label=label if i == 0 else "_")
ax2.axhline(SPEC, color="green", ls="--", lw=1.2, label=f"Spec ({SPEC*100:.0f} cm)")
ax2.set_title("Approach aligned by target-set time")
ax2.set_xlabel("Time since target set (s)"); ax2.set_ylabel("Error (m)")
ax2.set_ylim(bottom=0); ax2.set_xlim(left=0); ax2.legend(); ax2.grid(True, alpha=0.3)

# right: per-axis comparison (average approach, sim vs real)
ax3 = axes2[2]
for d, color, label, ls in [(d_real, "blue", "Real", "-"), (d_sim, "red", "Sim", "--")]:
    if d is None:
        continue
    segs = approach_segments(d)
    # interpolate to common time grid
    t_max = min(seg["time_s"][-1] for seg in segs)
    t_grid = np.linspace(0, t_max, 200)
    for axis_key, axis_label, alpha in [
        ("err_x", "x", 0.9), ("err_y", "y", 0.7), ("err_z", "z", 0.5)
    ]:
        interped = []
        for seg in segs:
            if len(seg["time_s"]) < 3: continue
            interped.append(np.interp(t_grid, seg["time_s"],
                                      np.abs(seg[axis_key])))
        if not interped: continue
        mean_err = np.mean(interped, axis=0)
        ax3.plot(t_grid, mean_err, color=color, ls=ls, lw=1.0, alpha=alpha,
                 label=f"{label} |{axis_key[-1]}|")

ax3.set_title("Mean per-axis (averaged over approaches)")
ax3.set_xlabel("Time since target set (s)"); ax3.set_ylabel("Error (m)")
ax3.set_ylim(bottom=0); ax3.set_xlim(left=0); ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("comparison_30cms.png", dpi=150)
print("Saved: comparison_30cms.png")


# ── 5. Summary bar chart ─────────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(10, 4))
fig3.suptitle("Mean ± Std Position Error by Speed (Real Drone)", fontsize=12)

speeds, means, stds = [], [], []
for label, fname in REAL_LOGS:
    d = load(fname)
    if d is None: continue
    trans = find_transitions(d["err_mag"])
    ends  = trans + [len(d["err_mag"])]
    all_vals = []
    for prev, end in zip([0] + trans, ends):
        seg = d["err_mag"][prev:end]
        t_seg = d["time_s"][prev:end] - d["time_s"][prev]
        mask = t_seg >= (t_seg[-1] - 2.0)  # last 2 s before target change
        all_vals.extend(seg[mask].tolist())
    speeds.append(label)
    means.append(np.mean(all_vals))
    stds.append(np.std(all_vals))

x = np.arange(len(speeds))
w = 0.35
b1 = ax.bar(x - w/2, means, w, label="Mean |err|", color="steelblue")
b2 = ax.bar(x + w/2, stds,  w, label="Std  |err|", color="tomato")
ax.axhline(SPEC, color="green", ls="--", lw=1.5, label=f"Spec ({SPEC*100:.0f} mm)")
ax.set_xticks(x); ax.set_xticklabels(speeds)
ax.set_ylabel("Error (m)"); ax.legend(); ax.grid(True, alpha=0.3, axis="y")

for b in b1:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
            f"{b.get_height()*100:.1f}cm", ha="center", va="bottom", fontsize=8)
for b in b2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002,
            f"{b.get_height()*100:.1f}cm", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("log_summary.png", dpi=150)
print("Saved: log_summary.png")


# ── 6. Console: transition timestamps for video sync ─────────────────────────
print("\n=== Target transition timestamps (sync with video) ===")
for label, fname in REAL_LOGS + [("SIM 30cm/s", "pos_error_sim_30.csv")]:
    d = load(fname)
    if d is None:
        print(f"\n{label}: not found"); continue
    t = d["time_s"] - d["time_s"][0]
    trans = find_transitions(d["err_mag"])
    print(f"\n{label} — transitions at t = {[round(float(t[i]),1) for i in trans]} s")
    for k, (s, e) in enumerate(zip([0]+trans, trans+[len(t)])):
        seg_mag = d["err_mag"][s:e]
        seg_t   = t[s:e]
        mask    = seg_t >= (seg_t[-1] - 2.0)
        mx = np.abs(d["err_x"][s:e][mask]).mean()
        my = np.abs(d["err_y"][s:e][mask]).mean()
        mz = np.abs(d["err_z"][s:e][mask]).mean()
        mt = seg_mag[mask].mean()
        print(f"  leg {k+1}: |x|={mx*100:.1f}cm  |y|={my*100:.1f}cm  "
              f"|z|={mz*100:.1f}cm  |total|={mt*100:.1f}cm")

plt.show()
