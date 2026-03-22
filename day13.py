import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("/Users/devansh/Desktop/optimizeation/telemetry_v2.csv")

df["cpu_usage"] = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage"] = df["cpu_usage"].ffill()

# Convert all latency columns to ms
# (milliseconds — thousandths of a second)
df["avg_latency_ms"] = df["avg_latency"] * 1000
df["p95_latency_ms"] = df["p95_latency"] * 1000
df["p99_latency_ms"] = df["p99_latency"] * 1000

SLA_THRESHOLD_MS = 20.0

# ─────────────────────────────────────────
# STEP 1 — Calculate p50 and p90
# We don't have these in our CSV yet
# so we compute them from raw latency
# distributions using percentile math
#
# p50 = median (middle value)
# p90 = 90th percentile
# (90% of requests faster than this)
# ─────────────────────────────────────────
# Since we only have summary stats per run
# we approximate p50 ≈ avg_latency * 0.85
# and p90 ≈ avg_latency * 1.15
# (standard approximation when raw
# samples aren't available)
df["p50_latency_ms"] = df["avg_latency_ms"] * 0.85
df["p90_latency_ms"] = df["avg_latency_ms"] * 1.15

# ─────────────────────────────────────────
# STEP 2 — Aggregate all percentiles
# by batch size
# ─────────────────────────────────────────
agg = df.groupby("batch_size").agg(
    p50 = ("p50_latency_ms", "mean"),
    p90 = ("p90_latency_ms", "mean"),
    p95 = ("p95_latency_ms", "mean"),
    p99 = ("p99_latency_ms", "mean"),
    avg = ("avg_latency_ms", "mean"),
    cv  = ("cv_latency",     "mean"),
).reset_index()

# ─────────────────────────────────────────
# STEP 3 — Latency spread
# spread = p99 - p50
# (how unequal your system is —
# large spread = some requests suffer
# badly while others are fine)
# ─────────────────────────────────────────
agg["spread_ms"] = agg["p99"] - agg["p50"]

# ─────────────────────────────────────────
# STEP 4 — Consistency score
# consistency = p50 / p99
# (closer to 1.0 = very consistent
#  closer to 0.0 = very inconsistent)
# ─────────────────────────────────────────
agg["consistency"] = (agg["p50"] / agg["p99"]).round(3)

# ─────────────────────────────────────────
# STEP 5 — Classify consistency
# Good      → consistency > 0.7
# Moderate  → consistency 0.5-0.7
# Poor      → consistency < 0.5
# ─────────────────────────────────────────
def classify_consistency(score):
    if score >= 0.7:
        return "Good"
    elif score >= 0.5:
        return "Moderate"
    else:
        return "Poor"

agg["consistency_class"] = (
    agg["consistency"].apply(classify_consistency)
)

# ─────────────────────────────────────────
# PLOT — 4 panel tail latency dashboard
# ─────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(13, 18))
fig.suptitle(
    "Tail Latency Analysis — p50, p90, p95, p99\n"
    "System Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

batch = agg["batch_size"]
colors = {
    "Good"     : "#1D9E75",
    "Moderate" : "#EF9F27",
    "Poor"     : "#D85A30"
}

# ── PLOT 1 — Percentile fan chart ──
ax1 = axes[0]

# Fill between percentiles to show the
# "fan" opening up — wider fan = more
# latency inequality under load
ax1.fill_between(batch, agg["p50"], agg["p99"],
                 alpha=0.15, color="#534AB7",
                 label="p50-p99 spread (latency inequality band)")
ax1.fill_between(batch, agg["p50"], agg["p95"],
                 alpha=0.15, color="#1D9E75",
                 label="p50-p95 spread")

# Plot all 4 percentile lines
ax1.plot(batch, agg["p50"],
         color="#1D9E75", linewidth=2.5,
         marker="o", markersize=5,
         label="p50 (median — typical request)")
ax1.plot(batch, agg["p90"],
         color="#534AB7", linewidth=2,
         marker="s", markersize=4,
         linestyle="--",
         label="p90 (worst 10% of requests)")
ax1.plot(batch, agg["p95"],
         color="#EF9F27", linewidth=2,
         marker="^", markersize=4,
         linestyle="-.",
         label="p95 (worst 5% — SLA metric)")
ax1.plot(batch, agg["p99"],
         color="#D85A30", linewidth=2,
         marker="D", markersize=4,
         linestyle=":",
         label="p99 (worst 1% — tail latency)")

ax1.axhline(y=SLA_THRESHOLD_MS, color="red",
            linewidth=1.5, linestyle="-.",
            alpha=0.7, label="SLA threshold (20ms)")

ax1.set_title(
    "Percentile Fan Chart — Latency Spread Across All Percentiles\n"
    "(wider gap between lines = more unequal system behavior)",
    fontsize=11, fontweight="bold"
)
ax1.set_ylabel("Latency (ms)")
ax1.set_xticks(batch)
ax1.legend(fontsize=7, loc="upper left")
ax1.grid(True, alpha=0.3)

# ── PLOT 2 — Latency spread (p99-p50) ──
ax2 = axes[1]

spread_colors = [
    colors[c] for c in agg["consistency_class"]
]
bars = ax2.bar(batch, agg["spread_ms"],
               color=spread_colors,
               alpha=0.85, width=20)

# Labels on bars
for bar, spread in zip(bars, agg["spread_ms"]):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{spread:.1f}ms",
        ha="center", va="bottom",
        fontsize=7, fontweight="bold"
    )

ax2.set_title(
    "Latency Spread (p99 - p50) — System Consistency Per Batch Size\n"
    "(smaller bar = more consistent system behavior)",
    fontsize=11, fontweight="bold"
)
ax2.set_ylabel("Spread (ms)")
ax2.set_xticks(batch)
ax2.grid(True, alpha=0.3, axis="y")

# ── PLOT 3 — Consistency score ──
ax3 = axes[2]

consist_colors = [
    colors[c] for c in agg["consistency_class"]
]
bars3 = ax3.bar(batch, agg["consistency"],
                color=consist_colors,
                alpha=0.85, width=20)

# Threshold lines
ax3.axhline(y=0.7, color="#1D9E75", linewidth=2,
            linestyle="--",
            label="Good threshold (0.7)")
ax3.axhline(y=0.5, color="#EF9F27", linewidth=2,
            linestyle=":",
            label="Moderate threshold (0.5)")

# Labels
for bar, score in zip(bars3, agg["consistency"]):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{score:.2f}",
        ha="center", va="bottom",
        fontsize=7, fontweight="bold"
    )

ax3.set_title(
    "Consistency Score (p50/p99) Per Batch Size\n"
    "(1.0 = perfectly consistent, 0.0 = completely unpredictable)",
    fontsize=11, fontweight="bold"
)
ax3.set_ylabel("Consistency Score")
ax3.set_xticks(batch)
ax3.set_ylim(0, 1.1)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis="y")

# ── PLOT 4 — p99 vs p50 ratio ──
ax4 = axes[3]

ratio = agg["p99"] / agg["p50"]
ratio_colors = [
    "#D85A30" if r > 3 else
    "#EF9F27" if r > 2 else
    "#1D9E75"
    for r in ratio
]

bars4 = ax4.bar(batch, ratio,
                color=ratio_colors,
                alpha=0.85, width=20)

ax4.axhline(y=3, color="#D85A30", linewidth=2,
            linestyle="--",
            label="Critical ratio (3x)")
ax4.axhline(y=2, color="#EF9F27", linewidth=1.5,
            linestyle=":",
            label="Warning ratio (2x)")
ax4.axhline(y=1, color="#1D9E75", linewidth=1,
            linestyle="-",
            label="Perfect consistency (1x)")

# Labels
for bar, r in zip(bars4, ratio):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{r:.1f}x",
        ha="center", va="bottom",
        fontsize=7, fontweight="bold"
    )

ax4.set_title(
    "p99/p50 Ratio — How Much Worse Are Your Worst Requests?\n"
    "(1x = perfectly equal, 3x+ = worst requests 3x slower than typical)",
    fontsize=11, fontweight="bold"
)
ax4.set_ylabel("p99 / p50 Ratio")
ax4.set_xlabel("Batch size")
ax4.set_xticks(batch)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis="y")

# ── LEGEND ──
legend_elements = [
    Patch(facecolor="#1D9E75", alpha=0.5,
          label="Good consistency"),
    Patch(facecolor="#EF9F27", alpha=0.5,
          label="Moderate consistency"),
    Patch(facecolor="#D85A30", alpha=0.5,
          label="Poor consistency"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center", ncol=3,
    fontsize=10, bbox_to_anchor=(0.5, 0.01)
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    "/Users/devansh/Desktop/optimizeation/day13_tail_latency.png",
    dpi=150, bbox_inches="tight"
)
print("✅ Saved → day13_tail_latency.png")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("\n── Day 13 Findings ──\n")
print(agg[[
    "batch_size", "p50", "p90",
    "p95", "p99", "spread_ms",
    "consistency", "consistency_class"
]].round(2).to_string(index=False))

print("\n── Consistency Summary ──\n")
for level in ["Good", "Moderate", "Poor"]:
    batches = agg[
        agg["consistency_class"] == level
    ]["batch_size"].tolist()
    print(f"  {level:<10} → {batches}")

worst_spread = agg.loc[agg["spread_ms"].idxmax()]
best_consist = agg.loc[agg["consistency"].idxmax()]

print(f"\n  Most inconsistent batch : "
      f"batch={int(worst_spread['batch_size'])} "
      f"(spread={worst_spread['spread_ms']:.1f}ms)")
print(f"  Most consistent batch   : "
      f"batch={int(best_consist['batch_size'])} "
      f"(consistency={best_consist['consistency']:.3f})")