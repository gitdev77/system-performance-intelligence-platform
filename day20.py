import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import json
from datetime import datetime

# ─────────────────────────────────────────
# DAY 20 — BOTTLENECK DETECTION LOGIC ENGINE
#
# GOAL: Make the system diagnose itself.
# Input  → raw telemetry CSV
# Output → one verdict per batch size:
#           bottleneck type, severity,
#           confidence, recommendation
#
# WHY THIS MATTERS:
# This is the bridge between Week 3 (analysis)
# and Week 4 (ML). The verdicts this engine
# produces become the LABELS (y) that our
# ML model will learn to predict on Day 22.
#
# Think of it like this:
# Week 3 = doctor running tests
# Day 20 = doctor writing the diagnosis
# Week 4 = training a junior doctor to
#           make the same diagnosis automatically
# ─────────────────────────────────────────

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_FILE   = "/Users/devansh/Desktop/optimizeation/telemetry_v2.csv"
OUTPUT_PNG  = "/Users/devansh/Desktop/optimizeation/day20_bottleneck_engine.png"
OUTPUT_JSON = "/Users/devansh/Desktop/optimizeation/day20_verdicts.json"
OUTPUT_CSV  = "/Users/devansh/Desktop/optimizeation/day20_verdicts.csv"

SLA_THRESHOLD_MS = 20.0

# ── DETECTION THRESHOLDS ──
# These are the rules our engine uses.
# Each number is derived from what we already
# learned in Days 7-19. Not random guesses.

THROUGHPUT_COLLAPSE_PCT   = 35.0  # if throughput dropped >35% from peak
LATENCY_GROWTH_FACTOR     = 1.8   # if latency is >1.8x the minimum seen
CPU_SPIKE_THRESHOLD       = 75.0  # if avg cpu > 75% at a batch size
HIGH_VARIANCE_CV          = 0.35  # if cv_latency > 0.35 = unstable
NONLINEAR_D2_MULTIPLIER   = 1.5   # second derivative threshold multiplier
SLA_BREACH_RATE_CRITICAL  = 50.0  # if >50% of runs breach SLA = critical

# ─────────────────────────────────────────
# STEP 1 — LOAD & PREPARE DATA
# ─────────────────────────────────────────
df = pd.read_csv(DATA_FILE)

# Clean CPU (same rolling median approach as Day 19)
df["cpu_usage"] = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage"] = (
    df["cpu_usage"]
    .rolling(window=5, min_periods=1, center=True)
    .median()
    .ffill().bfill()
)

df["avg_latency_ms"] = df["avg_latency"] * 1000
df["p95_latency_ms"] = df["p95_latency"] * 1000
df["p99_latency_ms"] = df["p99_latency"] * 1000

# ─────────────────────────────────────────
# STEP 2 — AGGREGATE PER BATCH SIZE
# (same pattern as every day before)
# ─────────────────────────────────────────
agg = df.groupby("batch_size").agg(
    avg_latency_ms  = ("avg_latency_ms",  "mean"),
    p95_latency_ms  = ("p95_latency_ms",  "mean"),
    p99_latency_ms  = ("p99_latency_ms",  "mean"),
    cpu_usage       = ("cpu_usage",       "mean"),
    throughput      = ("throughput",      "mean"),
    cv_latency      = ("cv_latency",      "mean"),
    sla_breach_rate = ("sla_breach",      "mean"),
).reset_index()

agg["sla_breach_pct"] = agg["sla_breach_rate"] * 100

# ─────────────────────────────────────────
# STEP 3 — COMPUTE DERIVED SIGNALS
#
# These are the inputs our rules will check.
# Each one captures a different type of
# system stress.
# ─────────────────────────────────────────

# Signal 1: Throughput collapse %
# How far has throughput fallen from its peak?
# Peak throughput = healthiest point (smallest batch).
# (pct_of_peak = 100% means no collapse at all)
peak_throughput = agg["throughput"].max()
agg["throughput_pct_of_peak"] = (
    agg["throughput"] / peak_throughput
) * 100
agg["throughput_collapse_pct"] = (
    100 - agg["throughput_pct_of_peak"]
)

# Signal 2: Latency growth factor
# How many times bigger is latency vs the minimum?
# 1.0 = same as baseline (healthy)
# 2.5 = 2.5x worse than best case (dangerous)
min_latency = agg["avg_latency_ms"].min()
agg["latency_growth_factor"] = (
    agg["avg_latency_ms"] / min_latency
)

# Signal 3: Second derivative of latency
# (how fast is the latency GROWTH itself accelerating?)
# High d2 = latency is not just growing, it's
# growing faster and faster = non-linear = danger
agg["d1"] = np.gradient(
    agg["avg_latency_ms"],
    agg["batch_size"]
)
agg["d2"] = np.gradient(
    agg["d1"],
    agg["batch_size"]
)
d2_mean = agg["d2"].mean()
d2_std  = agg["d2"].std()

# Signal 4: CPU stress flag
# Is CPU consistently elevated at this batch size?
agg["cpu_stressed"] = agg["cpu_usage"] > CPU_SPIKE_THRESHOLD

print("✅ Step 1-3 complete — signals computed\n")

# ─────────────────────────────────────────
# STEP 4 — THE DETECTION ENGINE
#
# This is the core of Day 20.
# A set of if-else rules that look at all
# signals together and produce a verdict.
#
# BOTTLENECK TYPES (in priority order):
# 1. throughput_collapse  — throughput dying fast
# 2. computation_scaling  — pure math complexity
# 3. cpu_saturation       — CPU is the wall
# 4. scheduling_noise     — OS scheduler chaos
# 5. healthy              — nothing wrong
#
# SEVERITY LEVELS:
# critical → immediate action needed
# high     → plan to fix soon
# medium   → monitor closely
# low      → within acceptable range
# healthy  → all good
#
# CONFIDENCE:
# How sure are we of this diagnosis?
# "high"   = multiple signals agree
# "medium" = one strong signal
# "low"    = borderline, could go either way
# ─────────────────────────────────────────

def detect_bottleneck(row):
    """
    Takes one row (one batch size's aggregated stats)
    and returns a diagnosis dictionary.
    """

    # Pull all signals for this batch size
    batch          = int(row["batch_size"])
    tput_collapse  = row["throughput_collapse_pct"]
    lat_growth     = row["latency_growth_factor"]
    cpu            = row["cpu_usage"]
    cv             = row["cv_latency"]
    sla_pct        = row["sla_breach_pct"]
    d2             = row["d2"]
    cpu_stressed   = row["cpu_stressed"]

    # Count how many "alarm" signals are firing
    # (more alarms firing = higher confidence in verdict)
    alarms = {
        "throughput_collapsing" : tput_collapse > THROUGHPUT_COLLAPSE_PCT,
        "latency_growing_fast"  : lat_growth > LATENCY_GROWTH_FACTOR,
        "cpu_elevated"          : cpu_stressed,
        "high_variance"         : cv > HIGH_VARIANCE_CV,
        "sla_frequently_breached": sla_pct > SLA_BREACH_RATE_CRITICAL,
        "nonlinear_acceleration": d2 > (d2_mean + NONLINEAR_D2_MULTIPLIER * d2_std),
    }

    n_alarms = sum(alarms.values())

    # ── RULE 1: HEALTHY ──
    # No alarms, low SLA breach, stable latency
    if n_alarms == 0 and sla_pct == 0:
        return {
            "bottleneck_type" : "healthy",
            "severity"        : "healthy",
            "confidence"      : "high",
            "n_alarms"        : n_alarms,
            "alarms"          : alarms,
            "recommendation"  : "System operating normally. "
                                "Maintain current batch size."
        }

    # ── RULE 2: THROUGHPUT COLLAPSE ──
    # Throughput falling fast AND latency rising.
    # This is the dominant pattern we found on Day 8
    # (r = -0.888 throughput-latency correlation).
    # Most likely cause: computation scaling with batch size.
    if (alarms["throughput_collapsing"] and
            alarms["latency_growing_fast"]):

        confidence = "high" if n_alarms >= 3 else "medium"
        severity   = "critical" if sla_pct > SLA_BREACH_RATE_CRITICAL \
                     else "high"

        return {
            "bottleneck_type" : "throughput_collapse",
            "severity"        : severity,
            "confidence"      : confidence,
            "n_alarms"        : n_alarms,
            "alarms"          : alarms,
            "recommendation"  : f"Throughput has collapsed "
                                f"{tput_collapse:.0f}% from peak. "
                                f"Cap batch size below this point "
                                f"or scale horizontally."
        }

    # ── RULE 3: COMPUTATION SCALING ──
    # Latency growing non-linearly (d2 high)
    # but CPU not necessarily spiking.
    # Cause: O(n²) matrix multiply complexity —
    # the workload itself is the bottleneck,
    # not a resource running out.
    if (alarms["nonlinear_acceleration"] and
            alarms["latency_growing_fast"] and
            not cpu_stressed):

        confidence = "high" if alarms["sla_frequently_breached"] \
                     else "medium"
        severity   = "high"

        return {
            "bottleneck_type" : "computation_scaling",
            "severity"        : severity,
            "confidence"      : confidence,
            "n_alarms"        : n_alarms,
            "alarms"          : alarms,
            "recommendation"  : "Latency growing non-linearly "
                                "with batch size. "
                                "Consider batching strategy changes "
                                "or algorithmic optimization."
        }

    # ── RULE 4: CPU SATURATION ──
    # CPU consistently elevated AND latency rising.
    # Note: our data showed this is RARE on Apple Silicon
    # (CPU never truly saturated in our experiments).
    # But the rule is needed for generalization.
    if cpu_stressed and alarms["latency_growing_fast"]:

        confidence = "high" if alarms["throughput_collapsing"] \
                     else "medium"
        severity   = "critical" if cpu > 90 else "high"

        return {
            "bottleneck_type" : "cpu_saturation",
            "severity"        : severity,
            "confidence"      : confidence,
            "n_alarms"        : n_alarms,
            "alarms"          : alarms,
            "recommendation"  : f"CPU at {cpu:.0f}% — approaching "
                                f"saturation. Add compute resources "
                                f"or reduce concurrency."
        }

    # ── RULE 5: SCHEDULING NOISE ──
    # High variance (cv > threshold) but no clear
    # resource bottleneck. Cause: OS scheduler is
    # randomly delaying compute tasks.
    # This was the "spike" pattern we found on Day 19
    # at batch=128 and batch=320.
    if alarms["high_variance"] and not cpu_stressed:

        confidence = "medium"
        severity   = "medium" if sla_pct < 50 else "high"

        return {
            "bottleneck_type" : "scheduling_noise",
            "severity"        : severity,
            "confidence"      : confidence,
            "n_alarms"        : n_alarms,
            "alarms"          : alarms,
            "recommendation"  : f"High latency variance (cv={cv:.2f}). "
                                f"System unpredictable at this batch size. "
                                f"Avoid for latency-sensitive workloads."
        }

    # ── RULE 6: WARNING (BORDERLINE) ──
    # Some alarms firing but doesn't fit a clean pattern.
    # System is degrading but root cause is unclear.
    if n_alarms >= 1:
        return {
            "bottleneck_type" : "degrading",
            "severity"        : "medium" if n_alarms == 1 else "high",
            "confidence"      : "low",
            "n_alarms"        : n_alarms,
            "alarms"          : alarms,
            "recommendation"  : "Performance degrading. "
                                "Monitor closely. "
                                "Investigate throughput and latency trends."
        }

    # ── DEFAULT: LOW RISK ──
    return {
        "bottleneck_type" : "low_risk",
        "severity"        : "low",
        "confidence"      : "medium",
        "n_alarms"        : n_alarms,
        "alarms"          : alarms,
        "recommendation"  : "Minor degradation. Within acceptable range."
    }

# ─────────────────────────────────────────
# STEP 5 — RUN ENGINE ON ALL BATCH SIZES
# ─────────────────────────────────────────
print("Running detection engine on all batch sizes...\n")

verdicts = []
for _, row in agg.iterrows():
    result = detect_bottleneck(row)
    verdicts.append({
    "batch_size"      : int(row["batch_size"]),
    "avg_latency_ms"  : round(row["avg_latency_ms"], 2),
    "throughput"      : round(row["throughput"], 2),
    "cpu_usage"       : round(row["cpu_usage"], 1),
    "cv_latency"      : round(row["cv_latency"], 3),
    "sla_breach_pct"  : round(row["sla_breach_pct"], 1),
    "latency_growth"  : round(row["latency_growth_factor"], 2),
    "tput_collapse_pct": round(row["throughput_collapse_pct"], 1),
    "n_alarms"        : result["n_alarms"],
    "bottleneck_type" : result["bottleneck_type"],
    "severity"        : result["severity"],
    "confidence"      : result["confidence"],
    "recommendation"  : result["recommendation"],
    "alarms"          : result["alarms"],
})
verdicts_df = pd.DataFrame(verdicts)

# ─────────────────────────────────────────
# STEP 6 — SAVE OUTPUTS
#
# Save as both CSV and JSON.
# CSV  → feeds directly into Day 22 ML training
# JSON → feeds into Day 29 structured logging
#        and Day 56 LLM layer
# ─────────────────────────────────────────
verdicts_df.to_csv(OUTPUT_CSV, index=False)

# Build JSON with metadata
output_json = {
    "generated_at"    : datetime.now().isoformat(),
    "engine_version"  : "v1.0",
    "data_source"     : DATA_FILE,
    "total_batch_sizes": len(verdicts_df),
    "thresholds_used" : {
        "throughput_collapse_pct" : THROUGHPUT_COLLAPSE_PCT,
        "latency_growth_factor"   : LATENCY_GROWTH_FACTOR,
        "cpu_spike_threshold"     : CPU_SPIKE_THRESHOLD,
        "high_variance_cv"        : HIGH_VARIANCE_CV,
        "sla_breach_rate_critical": SLA_BREACH_RATE_CRITICAL,
    },
    "verdicts": verdicts
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(output_json, f, indent=2, default=lambda x: int(x) if hasattr(x, 'item') else str(x))


print(f"✅ Verdicts saved → {OUTPUT_CSV}")
print(f"✅ Verdicts saved → {OUTPUT_JSON}\n")

# ─────────────────────────────────────────
# STEP 7 — VISUALIZE (3 panels)
# ─────────────────────────────────────────

# Color map for severity levels
# (same logic as a traffic light system)
severity_colors = {
    "healthy"  : "#1D9E75",
    "low"      : "#76C893",
    "medium"   : "#EF9F27",
    "high"     : "#E06C2A",
    "critical" : "#D85A30",
}

# Color map for bottleneck types
type_colors = {
    "healthy"             : "#1D9E75",
    "low_risk"            : "#76C893",
    "degrading"           : "#EF9F27",
    "scheduling_noise"    : "#534AB7",
    "computation_scaling" : "#E06C2A",
    "throughput_collapse" : "#D85A30",
    "cpu_saturation"      : "#C0392B",
}

fig, axes = plt.subplots(3, 1, figsize=(13, 16))
fig.suptitle(
    "Day 20 — Bottleneck Detection Logic Engine\n"
    "System Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

batch_arr = verdicts_df["batch_size"].values

# ── PANEL 1 — Severity timeline ──
#
# Shows severity level at each batch size
# as a colored bar. Like a traffic light
# across the whole batch range.
ax1 = axes[0]

bar_colors1 = [
    severity_colors[s] for s in verdicts_df["severity"]
]
bars1 = ax1.bar(
    batch_arr,
    verdicts_df["avg_latency_ms"],
    color=bar_colors1,
    alpha=0.85, width=20
)

ax1.axhline(
    y=SLA_THRESHOLD_MS,
    color="#E24B4A", linewidth=2,
    linestyle="-.", label="SLA threshold (20ms)"
)

# Annotate each bar with severity label
for bar, sev, batch in zip(
    bars1,
    verdicts_df["severity"],
    batch_arr
):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        sev.upper(),
        ha="center", va="bottom",
        fontsize=7, fontweight="bold",
        rotation=45
    )

# Build legend for severity colors
legend_patches = [
    mpatches.Patch(color=c, label=s.upper())
    for s, c in severity_colors.items()
]
ax1.legend(handles=legend_patches, fontsize=7,
           loc="upper left", ncol=5)

ax1.set_title(
    "Severity Level at Each Batch Size\n"
    "(bar height = avg latency, color = severity verdict)",
    fontsize=11, fontweight="bold"
)
ax1.set_ylabel("Avg Latency (ms)")
ax1.set_xticks(batch_arr)
ax1.grid(True, alpha=0.3, axis="y")

# ── PANEL 2 — Bottleneck type map ──
#
# Shows WHAT kind of bottleneck was detected
# at each batch size. Different color per type.
ax2 = axes[1]

bar_colors2 = [
    type_colors.get(t, "#999999")
    for t in verdicts_df["bottleneck_type"]
]
bars2 = ax2.bar(
    batch_arr,
    verdicts_df["tput_collapse_pct"],
    color=bar_colors2,
    alpha=0.85, width=20
)

# Label each bar with bottleneck type
for bar, btype, batch in zip(
    bars2,
    verdicts_df["bottleneck_type"],
    batch_arr
):
    label = btype.replace("_", "\n")
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        label,
        ha="center", va="bottom",
        fontsize=6.5, fontweight="bold"
    )

type_patches = [
    mpatches.Patch(color=c, label=t.replace("_", " "))
    for t, c in type_colors.items()
]
ax2.legend(handles=type_patches, fontsize=7,
           loc="upper left", ncol=3)

ax2.set_title(
    "Bottleneck Type Classification Per Batch Size\n"
    "(bar height = throughput collapse %, color = bottleneck type)",
    fontsize=11, fontweight="bold"
)
ax2.set_ylabel("Throughput Collapse (%)")
ax2.set_xticks(batch_arr)
ax2.grid(True, alpha=0.3, axis="y")

# ── PANEL 3 — Alarm count heatmap ──
#
# Shows how many alarms fired at each batch size
# across each alarm type. Like a monitoring
# dashboard — red cells = that alarm was firing.
ax3 = axes[2]

alarm_names = [
    "throughput_collapsing",
    "latency_growing_fast",
    "cpu_elevated",
    "high_variance",
    "sla_frequently_breached",
    "nonlinear_acceleration",
]

# Build matrix: rows=alarm types, cols=batch sizes
# 1 = alarm firing (red), 0 = alarm silent (green)
alarm_matrix = np.zeros(
    (len(alarm_names), len(batch_arr))
)
for col_i, v in enumerate(verdicts):
    for row_i, alarm_name in enumerate(alarm_names):
        alarm_matrix[row_i, col_i] = (
            1 if v["alarms"].get(alarm_name, False)
            else 0
        )

im = ax3.imshow(
    alarm_matrix,
    cmap="RdYlGn_r",   # red=1(firing), green=0(silent)
    aspect="auto",
    vmin=0, vmax=1
)

ax3.set_xticks(range(len(batch_arr)))
ax3.set_xticklabels(batch_arr, fontsize=8)
ax3.set_yticks(range(len(alarm_names)))
ax3.set_yticklabels(
    [a.replace("_", " ") for a in alarm_names],
    fontsize=8
)

# Add ON/OFF text in each cell
for row_i in range(len(alarm_names)):
    for col_i in range(len(batch_arr)):
        val = alarm_matrix[row_i, col_i]
        ax3.text(
            col_i, row_i,
            "ON" if val == 1 else "off",
            ha="center", va="center",
            fontsize=7, fontweight="bold",
            color="white" if val == 1 else "#555555"
        )

ax3.set_title(
    "Alarm State Heatmap — Which Signals Are Firing at Each Batch Size\n"
    "(red ON = alarm active | green off = signal normal)",
    fontsize=11, fontweight="bold"
)
ax3.set_xlabel("Batch size")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"✅ Chart saved → {OUTPUT_PNG}\n")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("=" * 65)
print("  DAY 20 FINDINGS — BOTTLENECK DETECTION ENGINE")
print("=" * 65)

print(f"\n── Verdict Table ──\n")
print(f"  {'Batch':<8} {'Type':<22} {'Severity':<10} "
      f"{'Conf':<8} {'Alarms':<8} {'SLA%'}")
print(f"  {'-'*8} {'-'*22} {'-'*10} {'-'*8} {'-'*8} {'-'*6}")

for _, row in verdicts_df.iterrows():
    print(
        f"  {int(row['batch_size']):<8} "
        f"{row['bottleneck_type']:<22} "
        f"{row['severity']:<10} "
        f"{row['confidence']:<8} "
        f"{int(row['n_alarms']):<8} "
        f"{row['sla_breach_pct']:.0f}%"
    )

# Summary counts
print(f"\n── Severity Summary ──\n")
for level in ["healthy", "low", "medium", "high", "critical"]:
    rows = verdicts_df[verdicts_df["severity"] == level]
    batches = rows["batch_size"].tolist()
    if batches:
        print(f"  {level.upper():<10} → "
              f"{len(batches)} batch sizes: {batches}")

print(f"\n── Bottleneck Type Summary ──\n")
type_counts = verdicts_df["bottleneck_type"].value_counts()
for btype, count in type_counts.items():
    batches = verdicts_df[
        verdicts_df["bottleneck_type"] == btype
    ]["batch_size"].tolist()
    print(f"  {btype:<25} → {count}x at batches {batches}")

print(f"\n── What This Means for Day 22 ML ──\n")
print(f"  These {len(verdicts_df)} verdicts are now your ML training labels.")
print(f"  The ML model will learn to predict 'bottleneck_type'")
print(f"  and 'severity' from raw telemetry features automatically.")
print(f"  Labels saved to → day20_verdicts.csv")
print(f"  JSON saved to   → day20_verdicts.json")

print(f"\n{'='*65}")
print(f"  Day 20 complete.")
print(f"{'='*65}\n")
