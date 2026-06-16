import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score,
                             f1_score)

# ─────────────────────────────────────────
# DAY 29 — FALSE POSITIVE TRACKING
# MLOps Layer, Day 3
#
# GOAL: Simulate multiple monitoring sessions
# with increasing data perturbation (noise),
# track how alert quality changes over time,
# and identify at what noise level the model
# starts degrading.
#
# This answers: "how robust is our monitoring
# system to real-world data drift?"
# ─────────────────────────────────────────

DATA_FILE  = "/Users/devansh/Desktop/optimizeation/telemetry_v2.csv"
TRAIN_FILE = "/Users/devansh/Desktop/optimizeation/day24_train.csv"
LOG_FILE   = "/Users/devansh/Desktop/optimizeation/day28_monitoring.jsonl"
OUTPUT_PNG = "/Users/devansh/Desktop/optimizeation/day29_fp_tracking.png"
OUTPUT_JSON= "/Users/devansh/Desktop/optimizeation/day29_session_tracker.json"

SLA_THRESHOLD_MS = 20.0
ALERT_THRESHOLD  = 0.7
MATRIX_DIM       = 512
BYTES_PER_FLOAT  = 4
N_SESSIONS       = 10   # simulate 10 monitoring sessions
RANDOM_SEED      = 42

# noise levels per session (0.0 = no noise,
# 0.1 = 10% std deviation noise added to features)
# increasing noise simulates data drift over time
NOISE_LEVELS = np.linspace(0.0, 0.25, N_SESSIONS)

# ─────────────────────────────────────────
# STEP 1 — Load & prepare base dataset
# ─────────────────────────────────────────
df = pd.read_csv(DATA_FILE)

df["cpu_usage_clean"] = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage_clean"] = (
    df["cpu_usage_clean"]
    .rolling(window=5, min_periods=1, center=True)
    .median().ffill().bfill()
)
df["avg_latency_ms"] = df["avg_latency"] * 1000
df["sla_breach_new"] = (
    df["avg_latency_ms"] > SLA_THRESHOLD_MS
).astype(int)

throughput_peak = (
    df.groupby("batch_size")["throughput"].mean().max()
)
df["rho"] = (
    1 - df["throughput"] / throughput_peak
).clip(lower=0)

b = df["batch_size"]
flops       = 2 * b * MATRIX_DIM * MATRIX_DIM
bytes_moved = ((b * MATRIX_DIM) +
               (MATRIX_DIM * MATRIX_DIM) +
               (b * MATRIX_DIM)) * BYTES_PER_FLOAT
df["AI"] = flops / bytes_moved

df_stream = df.sort_values(
    "batch_size"
).reset_index(drop=True)

print(f"✅ Step 1 — Base dataset prepared "
      f"({len(df_stream)} rows)\n")

# ─────────────────────────────────────────
# STEP 2 — Train model (same as Day 27/28)
# ─────────────────────────────────────────
train    = pd.read_csv(TRAIN_FILE)
FEATURES = ["batch_size", "cpu_usage_clean",
            "throughput", "cv_latency", "rho", "AI"]
TARGET   = "sla_breach_new"

scaler = StandardScaler()
X_tr   = scaler.fit_transform(train[FEATURES])
y_tr   = train[TARGET]

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)
model.fit(X_tr, y_tr)
print(f"✅ Step 2 — Model retrained\n")

# ─────────────────────────────────────────
# STEP 3 — MULTI-SESSION SIMULATION
#
# For each session:
# 1. Take the base dataset
# 2. Add a small amount of Gaussian noise
#    to the features (simulates real-world
#    variation between sessions)
# 3. Run the prediction loop
# 4. Record TP, FP, FN, TN, FPR, recall, F1
#
# "Gaussian noise" = random numbers drawn
# from a bell curve centered at 0.
# We multiply by noise_level × feature_std
# so the noise scales with the feature's
# natural variation (not a fixed amount).
# ─────────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)

session_records = []

print("Running multi-session simulation...\n")
print(f"  {'Session':<10} {'Noise':<8} {'TP':<5} "
      f"{'FP':<5} {'FN':<5} {'TN':<5} "
      f"{'FPR':<8} {'Recall':<8} {'F1'}")
print(f"  {'-'*10} {'-'*8} {'-'*5} {'-'*5} "
      f"{'-'*5} {'-'*5} {'-'*8} {'-'*8} {'-'*6}")

for session_idx in range(N_SESSIONS):
    noise_level = NOISE_LEVELS[session_idx]

    # Add noise to a copy of the feature columns
    # (never modify the original df)
    session_df = df_stream.copy()

    if noise_level > 0:
        for col in ["cpu_usage_clean", "throughput",
                    "cv_latency"]:
            col_std = session_df[col].std()
            noise   = rng.normal(
                loc=0,
                scale=noise_level * col_std,
                size=len(session_df)
            )
            session_df[col] = (
                session_df[col] + noise
            ).clip(lower=0)

        # Recompute rho after throughput is perturbed
        # (rho depends on throughput, so it changes too)
        session_df["rho"] = (
            1 - session_df["throughput"] / throughput_peak
        ).clip(lower=0, upper=0.97)

    # Run prediction loop
    X_session = scaler.transform(
        session_df[FEATURES]
    )
    probs   = model.predict_proba(X_session)[:, 1]
    preds   = (probs >= ALERT_THRESHOLD).astype(int)
    actuals = session_df[TARGET].values

    # Compute confusion counts
    tp = int(((preds == 1) & (actuals == 1)).sum())
    fp = int(((preds == 1) & (actuals == 0)).sum())
    fn = int(((preds == 0) & (actuals == 1)).sum())
    tn = int(((preds == 0) & (actuals == 0)).sum())

    fpr    = fp / (fp + tn) if (fp + tn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1     = (2 * prec * recall /
              (prec + recall)
              if (prec + recall) > 0 else 0)

    # Alert fatigue score:
    # "what fraction of ALL alerts were false alarms?"
    # 0.0 = no fatigue, 1.0 = all alerts were false
    total_alerts = tp + fp
    fatigue_score = (fp / total_alerts
                     if total_alerts > 0 else 0)

    record = {
        "session"       : session_idx + 1,
        "noise_level"   : round(float(noise_level), 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "fpr"           : round(fpr, 4),
        "recall"        : round(recall, 4),
        "precision"     : round(prec, 4),
        "f1"            : round(f1, 4),
        "fatigue_score" : round(fatigue_score, 4),
        "total_alerts"  : total_alerts,
        "timestamp"     : (
            datetime.now() + timedelta(days=session_idx)
        ).strftime("%Y-%m-%d"),
    }
    session_records.append(record)

    print(f"  Session {session_idx+1:<4} "
          f"{noise_level:<8.3f} "
          f"{tp:<5} {fp:<5} {fn:<5} {tn:<5} "
          f"{fpr:<8.3f} {recall:<8.3f} {f1:.3f}")

print()

# ─────────────────────────────────────────
# STEP 4 — LOAD DAY 28 LOG FOR REFERENCE
# (read existing log, extract baseline stats)
# ─────────────────────────────────────────
baseline_events = []
with open(LOG_FILE, "r") as f:
    for line in f:
        entry = json.loads(line.strip())
        if entry.get("log_type") == "INFERENCE_EVENT":
            baseline_events.append(entry)

baseline_probs = [
    e["model_output"]["breach_probability"]
    for e in baseline_events
]
baseline_outcomes = [e["outcome"] for e in baseline_events]

print(f"✅ Step 4 — Loaded {len(baseline_events)} "
      f"events from Day 28 log\n")

# ─────────────────────────────────────────
# STEP 5 — SAVE SESSION TRACKER
# ─────────────────────────────────────────
tracker_output = {
    "generated_at"    : datetime.now().isoformat(),
    "model_id"        : "sla_breach_classifier_v1",
    "alert_threshold" : ALERT_THRESHOLD,
    "n_sessions"      : N_SESSIONS,
    "noise_range"     : [0.0, 0.25],
    "sessions"        : session_records,
    "degradation_analysis": {
        "first_fp_session": next(
            (r["session"] for r in session_records
             if r["fp"] > 0), None
        ),
        "first_fn_session": next(
            (r["session"] for r in session_records
             if r["fn"] > 0), None
        ),
        "max_fpr": max(r["fpr"]
                       for r in session_records),
        "min_recall": min(r["recall"]
                          for r in session_records),
    }
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(tracker_output, f, indent=2)

print(f"✅ Step 5 — Session tracker saved "
      f"→ {OUTPUT_JSON}\n")

# ─────────────────────────────────────────
# STEP 6 — PLOT (4 panels)
# ─────────────────────────────────────────
sessions     = [r["session"]      for r in session_records]
noise_levels = [r["noise_level"]  for r in session_records]
fpr_vals     = [r["fpr"]          for r in session_records]
recall_vals  = [r["recall"]       for r in session_records]
f1_vals      = [r["f1"]           for r in session_records]
fp_vals      = [r["fp"]           for r in session_records]
fn_vals      = [r["fn"]           for r in session_records]
fatigue_vals = [r["fatigue_score"]for r in session_records]

colors = {
    "fpr"    : "#D85A30",
    "recall" : "#1D9E75",
    "f1"     : "#534AB7",
    "fp"     : "#EF9F27",
    "fn"     : "#D85A30",
    "fatigue": "#E24B4A",
    "noise"  : "#999999",
}

fig, axes = plt.subplots(4, 1, figsize=(13, 18))
fig.suptitle(
    "Day 29 — False Positive Tracking & Alert Quality\n"
    "System Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

# ── PANEL 1 — FPR + Recall over sessions ──
ax1 = axes[0]
ax1.plot(sessions, fpr_vals,
         color=colors["fpr"], linewidth=2.5,
         marker="o", markersize=7,
         label="False Positive Rate (FPR)")
ax1.plot(sessions, recall_vals,
         color=colors["recall"], linewidth=2.5,
         marker="s", markersize=7,
         label="Recall (breach detection rate)")
ax1.plot(sessions, f1_vals,
         color=colors["f1"], linewidth=2.5,
         marker="^", markersize=7,
         label="F1 Score")

ax1.axhline(y=1.0, color=colors["recall"],
            linewidth=1, linestyle=":",
            alpha=0.4)
ax1.axhline(y=0.0, color=colors["fpr"],
            linewidth=1, linestyle=":",
            alpha=0.4)

# Shade noise level on x-axis
ax1_twin = ax1.twinx()
ax1_twin.bar(sessions, noise_levels,
             color=colors["noise"],
             alpha=0.12, width=0.8)
ax1_twin.set_ylabel("Noise level added",
                    color=colors["noise"])
ax1_twin.tick_params(colors=colors["noise"])

ax1.set_title(
    "Alert Quality Over 10 Sessions — "
    "As Data Noise Increases\n"
    "(gray bars = noise level that session, "
    "simulating real-world data drift)",
    fontsize=11, fontweight="bold"
)
ax1.set_xlabel("Session number →")
ax1.set_ylabel("Metric value (0-1)")
ax1.set_ylim(-0.05, 1.15)
ax1.set_xticks(sessions)
ax1.set_xticklabels(
    [f"S{s}\n(noise={n:.2f})"
     for s, n in zip(sessions, noise_levels)],
    fontsize=7
)
ax1.legend(fontsize=8, loc="center right")
ax1.grid(True, alpha=0.3)

# ── PANEL 2 — FP and FN counts over sessions ──
ax2 = axes[1]
width = 0.35
x = np.arange(len(sessions))

bars_fp = ax2.bar(x - width/2, fp_vals, width,
                  color=colors["fp"],
                  alpha=0.85, label="FP (false alarms)")
bars_fn = ax2.bar(x + width/2, fn_vals, width,
                  color=colors["fn"],
                  alpha=0.85, label="FN (missed breaches)")

for bar, val in zip(bars_fp, fp_vals):
    if val > 0:
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 str(val), ha="center",
                 fontsize=9, fontweight="bold")
for bar, val in zip(bars_fn, fn_vals):
    if val > 0:
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 str(val), ha="center",
                 fontsize=9, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(
    [f"S{s}" for s in sessions], fontsize=8
)
ax2.set_title(
    "False Positive & False Negative Counts Per Session\n"
    "(FP = false alarms | FN = missed breaches — "
    "both should stay at 0)",
    fontsize=11, fontweight="bold"
)
ax2.set_ylabel("Count")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")

# ── PANEL 3 — Alert fatigue score ──
ax3 = axes[2]
fatigue_colors = [
    colors["recall"] if v == 0
    else colors["fp"] if v < 0.3
    else colors["fatigue"]
    for v in fatigue_vals
]
bars3 = ax3.bar(sessions, fatigue_vals,
                color=fatigue_colors,
                alpha=0.85, width=0.6)

for bar, val in zip(bars3, fatigue_vals):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f"{val:.2f}",
             ha="center", va="bottom",
             fontsize=9, fontweight="bold")

ax3.axhline(y=0.2, color=colors["fatigue"],
            linewidth=2, linestyle="--",
            label="Alert fatigue warning line (0.2)")

ax3.set_title(
    "Alert Fatigue Score Per Session\n"
    "(0 = no false alarms | >0.2 = engineers "
    "start ignoring alerts)",
    fontsize=11, fontweight="bold"
)
ax3.set_xlabel("Session number")
ax3.set_ylabel("Fatigue score\n(FP / total alerts)")
ax3.set_xticks(sessions)
ax3.set_ylim(0, 1.05)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis="y")

# ── PANEL 4 — Noise vs FPR scatter ──
# Shows the relationship between how much
# the world changed (noise) and how much
# alert quality degraded (FPR).
ax4 = axes[3]
scatter = ax4.scatter(
    noise_levels, fpr_vals,
    c=sessions, cmap="viridis",
    s=120, zorder=5,
    edgecolors="black", linewidth=0.5
)
plt.colorbar(scatter, ax=ax4,
             label="Session number")

for s, n, f in zip(sessions, noise_levels, fpr_vals):
    ax4.annotate(
        f" S{s}", (n, f), fontsize=8
    )

# Fit a trend line if there's variation
if max(fpr_vals) > 0:
    z = np.polyfit(noise_levels, fpr_vals, 1)
    trend_x = np.linspace(0, 0.25, 50)
    trend_y = z[0] * trend_x + z[1]
    ax4.plot(trend_x, trend_y,
             color=colors["fpr"],
             linewidth=2, linestyle="--",
             alpha=0.7,
             label="FPR trend with noise")
    ax4.legend(fontsize=8)

ax4.set_title(
    "Data Drift vs Alert Quality\n"
    "(x = how much data changed, "
    "y = false positive rate introduced)",
    fontsize=11, fontweight="bold"
)
ax4.set_xlabel("Noise level (simulated data drift)")
ax4.set_ylabel("False Positive Rate")
ax4.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"✅ Chart saved → {OUTPUT_PNG}\n")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
deg = tracker_output["degradation_analysis"]

print("=" * 65)
print("  DAY 29 FINDINGS — FALSE POSITIVE TRACKING")
print("=" * 65)

print(f"\n── Session-by-Session Results ──\n")
print(f"  {'S':<5} {'Noise':<8} {'FP':<5} {'FN':<5} "
      f"{'FPR':<8} {'Recall':<8} {'Fatigue'}")
print(f"  {'-'*5} {'-'*8} {'-'*5} {'-'*5} "
      f"{'-'*8} {'-'*8} {'-'*8}")
for r in session_records:
    flag = ""
    if r["fp"] > 0: flag += " ⚠️FP"
    if r["fn"] > 0: flag += " 🚨FN"
    print(f"  S{r['session']:<4} "
          f"{r['noise_level']:<8.3f} "
          f"{r['fp']:<5} {r['fn']:<5} "
          f"{r['fpr']:<8.4f} "
          f"{r['recall']:<8.4f} "
          f"{r['fatigue_score']:.4f}{flag}")

print(f"\n── Degradation Analysis ──\n")
if deg["first_fp_session"]:
    print(f"  First false positive appeared : "
          f"Session {deg['first_fp_session']}")
    noise_at_fp = session_records[
        deg["first_fp_session"] - 1
    ]["noise_level"]
    print(f"  Noise level at that point     : "
          f"{noise_at_fp:.3f} "
          f"({noise_at_fp*100:.1f}% perturbation)")
else:
    print(f"  ✅ No false positives across all "
          f"{N_SESSIONS} sessions")

if deg["first_fn_session"]:
    print(f"  First missed breach appeared  : "
          f"Session {deg['first_fn_session']}")
else:
    print(f"  ✅ No missed breaches across all "
          f"{N_SESSIONS} sessions")

print(f"  Max FPR across all sessions   : "
      f"{deg['max_fpr']:.4f}")
print(f"  Min Recall across all sessions: "
      f"{deg['min_recall']:.4f}")

print(f"\n── Robustness Verdict ──\n")
if deg["max_fpr"] == 0 and deg["min_recall"] == 1.0:
    print(f"  ✅ HIGHLY ROBUST — model maintains perfect")
    print(f"     precision and recall across all noise")
    print(f"     levels tested (0% to 25% perturbation).")
    print(f"     The breach signal is strong enough that")
    print(f"     small real-world variations don't confuse")
    print(f"     the model.")
elif deg["max_fpr"] < 0.05:
    print(f"  ✅ ROBUST — FPR stays below 5% even under")
    print(f"     significant data perturbation.")
else:
    print(f"  ⚠️  FRAGILE — model degrades under noise.")
    print(f"     Consider retraining more frequently or")
    print(f"     lowering the alert threshold.")

print(f"\n── What This Means For Production ──\n")
print(f"  If your system stays 'HIGHLY ROBUST':")
print(f"  → Retraining frequency: monthly is sufficient")
print(f"  → Alert threshold: keep at {ALERT_THRESHOLD}")
print(f"  → No drift detection system needed yet")
print(f"  If degradation appears at higher noise levels:")
print(f"  → Add a drift detection check (Day 35+)")
print(f"  → Consider MLflow model versioning (Day 30)")

print(f"\n{'='*65}")
print(f"  Day 29 complete.")
print(f"  → day29_fp_tracking.png")
print(f"  → day29_session_tracker.json")
print(f"{'='*65}\n")