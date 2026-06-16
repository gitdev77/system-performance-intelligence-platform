import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import json
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────
# DAY 27 — MONITORING SIMULATION LOOP
#
# GOAL: Simulate what happens when your trained
# model runs LIVE — receiving telemetry row by
# row, making predictions, firing alerts when
# breach probability crosses a threshold.
#
# Think of this as: "what does your system DO
# after it's deployed?" rather than "how do
# we train it?"
# ─────────────────────────────────────────

DATA_FILE  = "/Users/devansh/Desktop/optimizeation/telemetry_v2.csv"
TRAIN_FILE = "/Users/devansh/Desktop/optimizeation/day24_train.csv"
OUTPUT_PNG = "/Users/devansh/Desktop/optimizeation/day27_monitoring.png"
OUTPUT_LOG = "/Users/devansh/Desktop/optimizeation/day27_alert_log.json"

SLA_THRESHOLD_MS  = 20.0
MATRIX_DIM        = 512
BYTES_PER_FLOAT   = 4

# ── ALERT THRESHOLDS TO TEST ──
# We'll run the loop THREE times with different
# thresholds and compare the results.
# (threshold = "at what probability do we fire alert?")
THRESHOLDS = [0.3, 0.5, 0.7]

# ─────────────────────────────────────────
# STEP 1 — Load & prepare full dataset
# (same cleaning as Day 24)
# ─────────────────────────────────────────
df = pd.read_csv(DATA_FILE)

df["cpu_usage_clean"] = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage_clean"] = (
    df["cpu_usage_clean"]
    .rolling(window=5, min_periods=1, center=True)
    .median().ffill().bfill()
)
df["avg_latency_ms"] = df["avg_latency"] * 1000
df["sla_breach_new"] = (df["avg_latency_ms"] > SLA_THRESHOLD_MS).astype(int)

throughput_peak = df.groupby("batch_size")["throughput"].mean().max()
df["rho"] = (1 - df["throughput"] / throughput_peak).clip(lower=0)

b = df["batch_size"]
flops       = 2 * b * MATRIX_DIM * MATRIX_DIM
bytes_moved = ((b * MATRIX_DIM) + (MATRIX_DIM * MATRIX_DIM) +
               (b * MATRIX_DIM)) * BYTES_PER_FLOAT
df["AI"] = flops / bytes_moved

print(f"✅ Step 1 — Full dataset prepared ({len(df)} rows)\n")

# ─────────────────────────────────────────
# STEP 2 — Retrain Model B on ALL training data
#
# Day 25 trained on 83 rows (80% split).
# For the monitoring simulation, we retrain
# on the full train split from day24_train.csv
# (same data, just making sure model is fresh).
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

print(f"✅ Step 2 — Model B retrained on {len(train)} rows\n")

# ─────────────────────────────────────────
# STEP 3 — THE MONITORING LOOP
#
# This is the core of Day 27.
# We go through all 104 rows ONE AT A TIME,
# as if they're arriving in real time
# (we simulate timestamps 1 second apart).
#
# For each row:
#   1. Scale the features
#   2. Get breach probability from model
#   3. Compare to threshold
#   4. If above threshold → FIRE ALERT
#   5. Compare to actual outcome → TP/FP/FN/TN
#   6. Log everything
# ─────────────────────────────────────────

# Sort by batch_size so the "stream" goes from
# small to large batches (simulates ramping up load)
df_stream = df.sort_values("batch_size").reset_index(drop=True)

# Simulate timestamps starting from "now"
start_time = datetime(2026, 6, 16, 9, 0, 0)

def run_monitoring_loop(threshold):
    """
    Run the full prediction loop with a given
    alert threshold. Returns a list of event
    dictionaries (one per row).
    """
    events = []

    for i, row in df_stream.iterrows():
        # Simulate timestamp (1 row per second)
        timestamp = start_time + timedelta(seconds=i)

        # Build feature vector for this row
        features = pd.DataFrame([{
            "batch_size"      : row["batch_size"],
            "cpu_usage_clean" : row["cpu_usage_clean"],
            "throughput"      : row["throughput"],
            "cv_latency"      : row["cv_latency"],
            "rho"             : row["rho"],
            "AI"              : row["AI"],
        }])

        # Scale using the SAME scaler fitted on train
        features_scaled = scaler.transform(features)

        # Get probability of breach (the number between
        # 0 and 1 the model outputs)
        breach_prob = model.predict_proba(features_scaled)[0][1]

        # Fire alert if probability crosses threshold
        alert_fired = breach_prob >= threshold

        # Actual ground truth (did it REALLY breach?)
        actual_breach = bool(row["sla_breach_new"] == 1)

        # Label this event:
        # TP = alert fired AND real breach happened
        # FP = alert fired BUT no real breach (false alarm)
        # FN = no alert BUT real breach happened (missed!)
        # TN = no alert AND no real breach (correct silence)
        if alert_fired and actual_breach:
            outcome = "TP"
        elif alert_fired and not actual_breach:
            outcome = "FP"
        elif not alert_fired and actual_breach:
            outcome = "FN"
        else:
            outcome = "TN"

        events.append({
            "timestamp"    : timestamp.isoformat(),
            "run_index"    : int(i),
            "batch_size"   : int(row["batch_size"]),
            "cpu"          : round(float(row["cpu_usage_clean"]), 1),
            "throughput"   : round(float(row["throughput"]), 2),
            "cv_latency"   : round(float(row["cv_latency"]), 3),
            "rho"          : round(float(row["rho"]), 3),
            "AI"           : round(float(row["AI"]), 2),
            "breach_prob"  : round(float(breach_prob), 4),
            "threshold"    : threshold,
            "alert_fired"  : bool(alert_fired),
            "actual_breach": actual_breach,
            "outcome"      : outcome,
            "latency_ms"   : round(float(row["avg_latency_ms"]), 2),
        })

    return events

# Run for all three thresholds
print("Running monitoring loop for 3 thresholds...\n")
results = {}
for thresh in THRESHOLDS:
    events = run_monitoring_loop(thresh)
    results[thresh] = events
    tp = sum(1 for e in events if e["outcome"] == "TP")
    fp = sum(1 for e in events if e["outcome"] == "FP")
    fn = sum(1 for e in events if e["outcome"] == "FN")
    tn = sum(1 for e in events if e["outcome"] == "TN")
    print(f"  Threshold {thresh}: "
          f"TP={tp} FP={fp} FN={fn} TN={tn} "
          f"| Alerts fired={tp+fp}")

# ─────────────────────────────────────────
# STEP 4 — SAVE ALERT LOG (JSON)
#
# Save the threshold=0.5 run as a structured
# JSON log file. This is what a real monitoring
# system would write to disk continuously.
# Day 28 will build on this file directly.
# ─────────────────────────────────────────
log_output = {
    "simulation_metadata": {
        "generated_at"  : datetime.now().isoformat(),
        "model_version" : "logistic_regression_v1_physics_informed",
        "total_events"  : len(results[0.5]),
        "sla_threshold_ms": SLA_THRESHOLD_MS,
        "features_used" : FEATURES,
    },
    "threshold_runs": {
        str(t): results[t] for t in THRESHOLDS
    }
}

with open(OUTPUT_LOG, "w") as f:
    json.dump(log_output, f, indent=2)

print(f"\n✅ Alert log saved → {OUTPUT_LOG}\n")

# ─────────────────────────────────────────
# STEP 5 — PLOT (4 panels)
# ─────────────────────────────────────────
colors = {
    "prob"  : "#534AB7",
    "thresh": "#E24B4A",
    "tp"    : "#1D9E75",
    "fp"    : "#EF9F27",
    "fn"    : "#D85A30",
    "tn"    : "#999999",
    "actual": "#D85A30",
}

fig, axes = plt.subplots(4, 1, figsize=(14, 18))
fig.suptitle(
    "Day 27 — Monitoring Simulation Loop\n"
    "System Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

events_05 = results[0.5]  # use threshold=0.5 for panels 1-3
indices    = [e["run_index"]   for e in events_05]
probs      = [e["breach_prob"] for e in events_05]
actuals    = [e["actual_breach"] for e in events_05]
batches    = [e["batch_size"]  for e in events_05]

# ── PANEL 1 — Breach probability over time ──
# Shows the model's "confidence" score for each
# incoming run. Spikes = model sees danger.
ax1 = axes[0]
ax1.plot(indices, probs, color=colors["prob"],
         linewidth=1.5, alpha=0.8,
         label="Breach probability")
ax1.axhline(y=0.5, color=colors["thresh"],
            linewidth=2, linestyle="--",
            label="Alert threshold (0.5)")
ax1.fill_between(indices, probs, 0.5,
                 where=[p >= 0.5 for p in probs],
                 color=colors["thresh"], alpha=0.2,
                 label="Alert zone")

# Mark actual breaches with red X
for e in events_05:
    if e["actual_breach"]:
        ax1.axvline(x=e["run_index"],
                    color=colors["actual"],
                    linewidth=2, linestyle=":",
                    alpha=0.7)
        ax1.scatter(e["run_index"], e["breach_prob"],
                    color=colors["actual"], s=100,
                    zorder=5, marker="X")

ax1.set_title(
    "Model's Breach Probability — All 104 Runs (Streaming)\n"
    "(red X = actual breach, shaded = alert fired)",
    fontsize=11, fontweight="bold"
)
ax1.set_ylabel("P(breach)")
ax1.set_xlabel("Run index (time →)")
ax1.set_ylim(-0.05, 1.05)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# ── PANEL 2 — Outcome per run (color coded) ──
ax2 = axes[1]
outcome_colors = {
    "TP": colors["tp"],
    "FP": colors["fp"],
    "FN": colors["fn"],
    "TN": colors["tn"],
}
for e in events_05:
    ax2.bar(e["run_index"], 1,
            color=outcome_colors[e["outcome"]],
            alpha=0.7, width=1)

legend_patches = [
    mpatches.Patch(color=colors["tp"], label="TP — caught breach"),
    mpatches.Patch(color=colors["fp"], label="FP — false alarm"),
    mpatches.Patch(color=colors["fn"], label="FN — missed breach"),
    mpatches.Patch(color=colors["tn"], label="TN — correct silence"),
]
ax2.legend(handles=legend_patches, fontsize=8,
           loc="upper right", ncol=4)
ax2.set_title(
    "Monitoring Outcome Per Run (threshold=0.5)\n"
    "(every bar = one incoming telemetry row)",
    fontsize=11, fontweight="bold"
)
ax2.set_ylabel("Event")
ax2.set_xlabel("Run index (time →)")
ax2.set_yticks([])
ax2.grid(False)

# ── PANEL 3 — Threshold comparison ──
# Shows how TP/FP/FN counts change as you
# make the threshold stricter or looser
ax3 = axes[2]

thresh_labels = [str(t) for t in THRESHOLDS]
tp_counts = []
fp_counts = []
fn_counts = []

for thresh in THRESHOLDS:
    evs = results[thresh]
    tp_counts.append(sum(1 for e in evs if e["outcome"] == "TP"))
    fp_counts.append(sum(1 for e in evs if e["outcome"] == "FP"))
    fn_counts.append(sum(1 for e in evs if e["outcome"] == "FN"))

x = np.arange(len(THRESHOLDS))
width = 0.25

ax3.bar(x - width, tp_counts, width,
        color=colors["tp"], alpha=0.85, label="TP (caught)")
ax3.bar(x,         fp_counts, width,
        color=colors["fp"], alpha=0.85, label="FP (false alarm)")
ax3.bar(x + width, fn_counts, width,
        color=colors["fn"], alpha=0.85, label="FN (missed!)")

ax3.set_xticks(x)
ax3.set_xticklabels([f"Threshold = {t}" for t in THRESHOLDS])
ax3.set_title(
    "Threshold Sensitivity — How Threshold Choice Affects Alert Quality\n"
    "(lower threshold = more alerts, higher = fewer but may miss breaches)",
    fontsize=11, fontweight="bold"
)
ax3.set_ylabel("Count")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis="y")

for i, (tp, fp, fn) in enumerate(zip(tp_counts, fp_counts, fn_counts)):
    ax3.text(i - width, tp + 0.05, str(tp),
             ha="center", fontsize=9, fontweight="bold")
    ax3.text(i,         fp + 0.05, str(fp),
             ha="center", fontsize=9, fontweight="bold")
    ax3.text(i + width, fn + 0.05, str(fn),
             ha="center", fontsize=9, fontweight="bold")

# ── PANEL 4 — Probability distribution ──
# Shows: are breach and no-breach runs clearly
# separated in probability space?
# Well-separated = reliable model.
# Overlapping = confused model.
ax4 = axes[3]

probs_breach    = [e["breach_prob"] for e in events_05
                   if e["actual_breach"]]
probs_nobreach  = [e["breach_prob"] for e in events_05
                   if not e["actual_breach"]]

ax4.hist(probs_nobreach, bins=20, alpha=0.7,
         color=colors["tn"], label="No-breach runs",
         density=True)
ax4.hist(probs_breach, bins=10, alpha=0.9,
         color=colors["actual"], label="Breach runs",
         density=True)
ax4.axvline(x=0.5, color=colors["thresh"],
            linewidth=2, linestyle="--",
            label="Threshold (0.5)")

ax4.set_title(
    "Probability Distribution — Breach vs No-Breach Runs\n"
    "(well-separated peaks = model is confident and correct)",
    fontsize=11, fontweight="bold"
)
ax4.set_xlabel("P(breach) assigned by model")
ax4.set_ylabel("Density")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)


plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"✅ Chart saved → {OUTPUT_PNG}\n")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("=" * 65)
print("  DAY 27 FINDINGS — MONITORING SIMULATION")
print("=" * 65)

print(f"\n── Threshold Sensitivity ──\n")
print(f"  {'Threshold':<12} {'Alerts':<10} {'TP':<6} "
      f"{'FP':<6} {'FN':<6} {'Recall':<10} {'Precision'}")
print(f"  {'-'*12} {'-'*10} {'-'*6} {'-'*6} {'-'*6} "
      f"{'-'*10} {'-'*10}")

total_actual_breaches = sum(1 for e in events_05
                            if e["actual_breach"])

for thresh in THRESHOLDS:
    evs  = results[thresh]
    tp   = sum(1 for e in evs if e["outcome"] == "TP")
    fp   = sum(1 for e in evs if e["outcome"] == "FP")
    fn   = sum(1 for e in evs if e["outcome"] == "FN")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"  {thresh:<12} {tp+fp:<10} {tp:<6} "
          f"{fp:<6} {fn:<6} {rec:<10.2f} {prec:.2f}")

print(f"\n  Total actual breaches in stream: "
      f"{total_actual_breaches}")

print(f"\n── Recommended Threshold ──\n")
best_thresh = None
best_recall = -1
for thresh in THRESHOLDS:
    evs = results[thresh]
    tp  = sum(1 for e in evs if e["outcome"] == "TP")
    fn  = sum(1 for e in evs if e["outcome"] == "FN")
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    if rec > best_recall:
        best_recall  = rec
        best_thresh  = thresh

print(f"  Best recall achieved at threshold = {best_thresh}")
print(f"  Recall = {best_recall:.2f} "
      f"(caught {best_recall*100:.0f}% of real breaches)")
print(f"  → For production SLA monitoring, lower thresholds")
print(f"    are preferred — missing a breach costs more")
print(f"    than a false alarm.")

print(f"\n── What Day 28 Will Do ──\n")
print(f"  Build structured JSON logging on top of this loop.")
print(f"  Instead of one big dump, Day 28 writes one log")
print(f"  entry per event — as if a real service is running.")
print(f"  This feeds directly into Day 29's false positive")
print(f"  tracker and eventually the Grafana dashboard.")

print(f"\n{'='*65}")
print(f"  Day 27 complete.")
print(f"  → day27_monitoring.png")
print(f"  → day27_alert_log.json ({len(results[0.5])} events logged)")
print(f"{'='*65}\n")