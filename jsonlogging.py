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

# ─────────────────────────────────────────
# DAY 28 — STRUCTURED JSON LOGGING
# MLOps Layer, Day 2
#
# GOAL: Replace the "big dump at the end"
# approach from Day 27 with proper structured
# logging — one JSON line written per event,
# as it happens, with severity levels.
#
# This is what separates a data science script
# from a production monitoring service.
# ─────────────────────────────────────────

DATA_FILE  = "/Users/devansh/Desktop/optimizeation/telemetry_v2.csv"
TRAIN_FILE = "/Users/devansh/Desktop/optimizeation/day24_train.csv"
OUTPUT_PNG  = "/Users/devansh/Desktop/optimizeation/day28_logging.png"
OUTPUT_JSONL = "/Users/devansh/Desktop/optimizeation/day28_monitoring.jsonl"
OUTPUT_SUMMARY = "/Users/devansh/Desktop/optimizeation/day28_session_summary.json"

SLA_THRESHOLD_MS = 20.0
ALERT_THRESHOLD  = 0.7   # best threshold from Day 27
MATRIX_DIM       = 512
BYTES_PER_FLOAT  = 4

# ─────────────────────────────────────────
# STEP 1 — Load & prepare (same as Day 27)
# ─────────────────────────────────────────
df = pd.read_csv(DATA_FILE)

df["cpu_usage_clean"] = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage_clean"] = (
    df["cpu_usage_clean"]
    .rolling(window=5, min_periods=1, center=True)
    .median().ffill().bfill()
)
df["avg_latency_ms"]  = df["avg_latency"] * 1000
df["sla_breach_new"]  = (
    df["avg_latency_ms"] > SLA_THRESHOLD_MS
).astype(int)

throughput_peak = df.groupby("batch_size")["throughput"].mean().max()
df["rho"] = (1 - df["throughput"] / throughput_peak).clip(lower=0)

b = df["batch_size"]
flops       = 2 * b * MATRIX_DIM * MATRIX_DIM
bytes_moved = ((b * MATRIX_DIM) + (MATRIX_DIM * MATRIX_DIM) +
               (b * MATRIX_DIM)) * BYTES_PER_FLOAT
df["AI"] = flops / bytes_moved

df_stream = df.sort_values("batch_size").reset_index(drop=True)

# ─────────────────────────────────────────
# STEP 2 — Retrain model (same as Day 27)
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

print(f"✅ Steps 1-2 — Data prepared, model retrained\n")

# ─────────────────────────────────────────
# STEP 3 — DETERMINE LOG LEVEL
#
# Every event gets a level based on what
# the model sees AND what actually happened.
# ─────────────────────────────────────────
def get_log_level(breach_prob, alert_fired, actual_breach):
    """
    Assign a severity level to each event.

    CRITICAL → actual breach confirmed
    ALERT    → model predicted breach
                (whether right or wrong)
    WARNING  → probability is elevated
                but below alert threshold
                (early warning zone)
    INFO     → everything normal
    """
    if actual_breach:
        return "CRITICAL"
    elif alert_fired:
        return "ALERT"
    elif breach_prob >= 0.4:
        # (elevated but not yet triggering alert
        #  — "keep watching this" zone)
        return "WARNING"
    else:
        return "INFO"

# ─────────────────────────────────────────
# STEP 4 — THE STRUCTURED LOGGING LOOP
#
# Key difference from Day 27:
# Instead of collecting everything in memory
# and saving at the end, we OPEN the log file
# and write each line as it's generated.
#
# "with open(..., 'w') as log_file:"
# keeps the file open for the whole loop,
# writing one line per event.
# ─────────────────────────────────────────

# Model metadata — written once at the top
model_metadata = {
    "model_id"       : "sla_breach_classifier_v1",
    "model_type"     : "LogisticRegression",
    "features"       : FEATURES,
    "trained_on_rows": len(train),
    "alert_threshold": ALERT_THRESHOLD,
    "sla_threshold_ms": SLA_THRESHOLD_MS,
    "physics_features": ["rho", "AI"],
    "training_date"  : datetime.now().strftime("%Y-%m-%d"),
    "project"        : "System Performance Intelligence Platform",
}

start_time = datetime(2026, 6, 16, 9, 0, 0)

# Counters for session summary
counts = {"INFO": 0, "WARNING": 0, "ALERT": 0, "CRITICAL": 0}
tp = fp = fn = tn = 0

# Collect for plotting
log_events = []

print("Writing structured log (one line per event)...\n")

with open(OUTPUT_JSONL, "w") as log_file:

    # Write metadata as first line (a comment-style header)
    header = {
        "log_type"  : "SESSION_START",
        "timestamp" : start_time.isoformat(),
        "metadata"  : model_metadata
    }
    log_file.write(json.dumps(header) + "\n")

    # Main event loop
    for i, row in df_stream.iterrows():
        timestamp = start_time + timedelta(seconds=i)

        # Build feature vector
        features = pd.DataFrame([{
            "batch_size"      : row["batch_size"],
            "cpu_usage_clean" : row["cpu_usage_clean"],
            "throughput"      : row["throughput"],
            "cv_latency"      : row["cv_latency"],
            "rho"             : row["rho"],
            "AI"              : row["AI"],
        }])

        features_scaled = scaler.transform(features)
        breach_prob     = model.predict_proba(
            features_scaled
        )[0][1]
        alert_fired     = breach_prob >= ALERT_THRESHOLD
        actual_breach   = bool(row["sla_breach_new"] == 1)

        # Outcome label
        if alert_fired and actual_breach:
            outcome = "TP"; tp += 1
        elif alert_fired and not actual_breach:
            outcome = "FP"; fp += 1
        elif not alert_fired and actual_breach:
            outcome = "FN"; fn += 1
        else:
            outcome = "TN"; tn += 1

        # Log level
        level = get_log_level(breach_prob, alert_fired,
                              actual_breach)
        counts[level] += 1

        # Build the log entry
        # (every field has a clear name — this is
        #  what makes it "structured" — machines
        #  can parse this reliably)
        log_entry = {
            "log_type"      : "INFERENCE_EVENT",
            "level"         : level,
            "timestamp"     : timestamp.isoformat(),
            "run_index"     : int(i),
            "batch_size"    : int(row["batch_size"]),
            "features": {
                "cpu"        : round(float(
                    row["cpu_usage_clean"]), 1),
                "throughput" : round(float(
                    row["throughput"]), 2),
                "cv_latency" : round(float(
                    row["cv_latency"]), 3),
                "rho"        : round(float(row["rho"]), 3),
                "AI"         : round(float(row["AI"]), 2),
            },
            "model_output": {
                "breach_probability" : round(float(
                    breach_prob), 4),
                "alert_fired"        : bool(alert_fired),
                "threshold_used"     : ALERT_THRESHOLD,
            },
            "ground_truth": {
                "actual_breach"  : actual_breach,
                "latency_ms"     : round(float(
                    row["avg_latency_ms"]), 2),
                "sla_threshold_ms": SLA_THRESHOLD_MS,
            },
            "outcome" : outcome,
        }

        # Write immediately — not at the end
        log_file.write(json.dumps(log_entry) + "\n")

        # Collect for plotting
        log_events.append(log_entry)

    # Write session end marker
    footer = {
        "log_type" : "SESSION_END",
        "timestamp": (start_time + timedelta(
            seconds=len(df_stream))).isoformat(),
        "total_events": len(df_stream),
    }
    log_file.write(json.dumps(footer) + "\n")

print(f"✅ Step 4 — {len(log_events)} events written "
      f"to {OUTPUT_JSONL}\n")

# ─────────────────────────────────────────
# STEP 5 — SESSION SUMMARY
#
# End-of-session report: overall stats,
# saved as a separate clean JSON file.
# Think of this as the "shift report" a
# monitoring engineer reads each morning.
# ─────────────────────────────────────────
precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_val    = tp / (tp + fn) if (tp + fn) > 0 else 0
fp_rate       = fp / (fp + tn) if (fp + tn) > 0 else 0
f1_val        = (2 * precision_val * recall_val /
                 (precision_val + recall_val)
                 if (precision_val + recall_val) > 0 else 0)

session_summary = {
    "session_id"       : "session_20260616_090000",
    "generated_at"     : datetime.now().isoformat(),
    "model_id"         : model_metadata["model_id"],
    "alert_threshold"  : ALERT_THRESHOLD,
    "total_events"     : len(log_events),
    "log_level_counts" : counts,
    "confusion_matrix" : {
        "TP": tp, "FP": fp,
        "FN": fn, "TN": tn
    },
    "metrics": {
        "precision"       : round(precision_val, 4),
        "recall"          : round(recall_val, 4),
        "f1_score"        : round(f1_val, 4),
        "false_positive_rate": round(fp_rate, 4),
    },
    "health_verdict": (
        "HEALTHY — model catching all breaches "
        "with zero false positives"
        if fp == 0 and fn == 0 else
        "DEGRADED — review false positives/negatives"
    ),
    "log_file": OUTPUT_JSONL,
}

with open(OUTPUT_SUMMARY, "w") as f:
    json.dump(session_summary, f, indent=2)

print(f"✅ Step 5 — Session summary saved "
      f"→ {OUTPUT_SUMMARY}\n")

# ─────────────────────────────────────────
# STEP 6 — PLOT (3 panels)
# ─────────────────────────────────────────
level_colors = {
    "INFO"    : "#999999",
    "WARNING" : "#EF9F27",
    "ALERT"   : "#534AB7",
    "CRITICAL": "#D85A30",
}

fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle(
    "Day 28 — Structured JSON Logging Layer\n"
    "System Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

indices = [e["run_index"] for e in log_events]
probs   = [e["model_output"]["breach_probability"]
           for e in log_events]
levels  = [e["level"] for e in log_events]

# ── PANEL 1 — Log level timeline ──
# Shows every event as a colored dot by level.
# At a glance: where are the warnings/alerts?
ax1 = axes[0]

level_y = {"INFO": 1, "WARNING": 2, "ALERT": 3, "CRITICAL": 4}
for e in log_events:
    ax1.scatter(
        e["run_index"],
        level_y[e["level"]],
        color=level_colors[e["level"]],
        s=60, alpha=0.8, zorder=5
    )

ax1.set_yticks([1, 2, 3, 4])
ax1.set_yticklabels(["INFO", "WARNING", "ALERT", "CRITICAL"])
ax1.set_title(
    "Log Level Timeline — Every Event Classified\n"
    "(scroll down the y-axis = escalating severity)",
    fontsize=11, fontweight="bold"
)
ax1.set_xlabel("Run index (time →)")
ax1.set_ylabel("Log Level")
ax1.grid(True, alpha=0.3, axis="x")

legend_patches = [
    mpatches.Patch(color=c, label=l)
    for l, c in level_colors.items()
]
ax1.legend(handles=legend_patches, fontsize=8,
           loc="upper left", ncol=4)

# ── PANEL 2 — Log level distribution bar ──
ax2 = axes[1]
level_order = ["INFO", "WARNING", "ALERT", "CRITICAL"]
level_counts_plot = [counts[l] for l in level_order]
bar_colors2 = [level_colors[l] for l in level_order]

bars2 = ax2.bar(level_order, level_counts_plot,
                color=bar_colors2, alpha=0.85, width=0.5)

for bar, val in zip(bars2, level_counts_plot):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        str(val),
        ha="center", va="bottom",
        fontsize=12, fontweight="bold"
    )

ax2.set_title(
    "Log Level Distribution — How Often Each Level Fired\n"
    "(INFO = normal | WARNING = watch | "
    "ALERT = predicted breach | CRITICAL = confirmed breach)",
    fontsize=11, fontweight="bold"
)
ax2.set_ylabel("Event count")
ax2.grid(True, alpha=0.3, axis="y")

# Add percentage labels
total_events = len(log_events)
for i, (level, cnt) in enumerate(
    zip(level_order, level_counts_plot)
):
    pct = (cnt / total_events) * 100
    ax2.text(i, cnt / 2, f"{pct:.1f}%",
             ha="center", va="center",
             fontsize=10, color="white",
             fontweight="bold")

# ── PANEL 3 — Breach probability with level bands ──
ax3 = axes[2]

# Color background bands for each level zone
ax3.axhspan(0.0, 0.4,  alpha=0.08,
            color=level_colors["INFO"],
            label="INFO zone (0 - 0.4)")
ax3.axhspan(0.4, 0.7,  alpha=0.08,
            color=level_colors["WARNING"],
            label="WARNING zone (0.4 - 0.7)")
ax3.axhspan(0.7, 1.0,  alpha=0.08,
            color=level_colors["ALERT"],
            label="ALERT / CRITICAL zone (>0.7)")

ax3.plot(indices, probs,
         color="#222222", linewidth=1.2,
         alpha=0.6, zorder=3)

for e in log_events:
    ax3.scatter(
        e["run_index"],
        e["model_output"]["breach_probability"],
        color=level_colors[e["level"]],
        s=30, alpha=0.9, zorder=5
    )

ax3.axhline(y=ALERT_THRESHOLD,
            color=level_colors["ALERT"],
            linewidth=2, linestyle="--",
            label=f"Alert threshold ({ALERT_THRESHOLD})")

ax3.set_title(
    "Breach Probability — Colored By Log Level\n"
    "(each dot color = level assigned to that event)",
    fontsize=11, fontweight="bold"
)
ax3.set_xlabel("Run index (time →)")
ax3.set_ylabel("P(breach)")
ax3.set_ylim(-0.05, 1.05)
ax3.legend(fontsize=8, loc="upper left")
ax3.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"✅ Chart saved → {OUTPUT_PNG}\n")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("=" * 65)
print("  DAY 28 FINDINGS — STRUCTURED LOGGING")
print("=" * 65)

print(f"\n── Log Level Distribution ──\n")
for level in level_order:
    pct = (counts[level] / total_events) * 100
    bar = "█" * int(pct / 2)
    print(f"  {level:<10} {counts[level]:>4} events "
          f"({pct:5.1f}%)  {bar}")

print(f"\n── Session Performance ──\n")
print(f"  Precision          : {precision_val:.3f}")
print(f"  Recall             : {recall_val:.3f}")
print(f"  F1 Score           : {f1_val:.3f}")
print(f"  False Positive Rate: {fp_rate:.3f}")
print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")

print(f"\n── Log Files Written ──\n")
print(f"  JSONL log  : {OUTPUT_JSONL}")
print(f"  Summary    : {OUTPUT_SUMMARY}")
print(f"  Total lines: {total_events + 2} "
      f"(104 events + header + footer)")

print(f"\n── Sample Log Entry (most recent CRITICAL) ──\n")
critical_events = [e for e in log_events
                   if e["level"] == "CRITICAL"]
if critical_events:
    sample = critical_events[-1]
    print(json.dumps(sample, indent=4))

print(f"\n── Health Verdict ──\n")
print(f"  {session_summary['health_verdict']}")

print(f"\n── What This Means For Your Project ──\n")
print(f"  day28_monitoring.jsonl is a real production-style")
print(f"  log file. Every line is independently parseable.")
print(f"  Day 29 reads this file to compute false positive")
print(f"  rates over time. Day 56's LLM layer reads this")
print(f"  file to generate natural language reports.")

print(f"\n{'='*65}")
print(f"  Day 28 complete.")
print(f"{'='*65}\n")