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
df["avg_latency_ms"] = df["avg_latency"] * 1000
df["p95_latency_ms"] = df["p95_latency"] * 1000
df["p99_latency_ms"] = df["p99_latency"] * 1000

# ─────────────────────────────────────────
# AGGREGATE by batch size
# ─────────────────────────────────────────
agg = df.groupby("batch_size").agg(
    avg_latency_ms = ("avg_latency_ms",  "mean"),
    p95_latency_ms = ("p95_latency_ms",  "mean"),
    p99_latency_ms = ("p99_latency_ms",  "mean"),
    latency_std    = ("latency_std",     "mean"),
    cv_latency     = ("cv_latency",      "mean"),
    cpu_usage      = ("cpu_usage",       "mean"),
    throughput     = ("throughput",      "mean"),
    sla_breach     = ("sla_breach",      "mean"),
).reset_index()

SLA_THRESHOLD_MS = 20.0

# ─────────────────────────────────────────
# STEP 1 — Compute Performance Risk Index
# PRI combines 4 signals into one score:
#
# Signal 1 — SLA proximity
# (how close p95 is to the 20ms limit)
# 0 = far from limit, 1 = at/past limit
#
# Signal 2 — Latency growth rate
# (how fast avg latency is growing vs
# the minimum latency in the dataset)
#
# Signal 3 — Tail risk
# (gap between p99 and avg — large gap
# means your worst requests are very bad)
#
# Signal 4 — Throughput collapse
# (how much throughput dropped vs peak)
# ─────────────────────────────────────────

# Signal 1 — SLA proximity (0 to 1)
agg["sla_proximity"] = (
    agg["p95_latency_ms"] / SLA_THRESHOLD_MS
).clip(0, 1)

# Signal 2 — Latency growth rate (0 to 1)
min_lat = agg["avg_latency_ms"].min()
max_lat = agg["avg_latency_ms"].max()
agg["latency_growth_rate"] = (
    (agg["avg_latency_ms"] - min_lat) /
    (max_lat - min_lat)
)

# Signal 3 — Tail risk (0 to 1)
# p99 - avg normalized by avg
# (how much worse your worst requests
# are compared to typical requests)
agg["tail_risk"] = (
    (agg["p99_latency_ms"] - agg["avg_latency_ms"]) /
    agg["avg_latency_ms"]
).clip(0, 1)

# Signal 4 — Throughput collapse (0 to 1)
max_tput = agg["throughput"].max()
agg["throughput_collapse"] = (
    1 - (agg["throughput"] / max_tput)
).clip(0, 1)

# ─────────────────────────────────────────
# STEP 2 — Combine into PRI score
# Weighted average of 4 signals
# Weights reflect importance:
# SLA proximity = most important (0.35)
# Latency growth = important (0.30)
# Tail risk = moderate (0.20)
# Throughput collapse = supporting (0.15)
# ─────────────────────────────────────────
agg["PRI"] = (
    0.35 * agg["sla_proximity"] +
    0.30 * agg["latency_growth_rate"] +
    0.20 * agg["tail_risk"] +
    0.15 * agg["throughput_collapse"]
)

# Normalize PRI to 0-100 for readability
agg["PRI_score"] = (
    agg["PRI"] /
    agg["PRI"].max() * 100
).round(1)

# ─────────────────────────────────────────
# STEP 3 — Risk classification
# Safe     → PRI_score < 40
# Caution  → PRI_score 40-70
# Critical → PRI_score > 70
# ─────────────────────────────────────────
def classify_risk(score):
    if score >= 70:
        return "Critical"
    elif score >= 40:
        return "Caution"
    else:
        return "Safe"

agg["risk_level"] = agg["PRI_score"].apply(classify_risk)

# ─────────────────────────────────────────
# STEP 4 — Identify cliff point
# Batch size with highest PRI score
# ─────────────────────────────────────────
cliff_idx   = agg["PRI_score"].idxmax()
cliff_batch = agg.loc[cliff_idx, "batch_size"]
cliff_score = agg.loc[cliff_idx, "PRI_score"]
cliff_lat   = agg.loc[cliff_idx, "avg_latency_ms"]

# ─────────────────────────────────────────
# PLOT — 3 panel risk dashboard
# ─────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 14))
fig.suptitle(
    "Performance Risk Index Dashboard\nSystem Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

risk_colors = {
    "Safe"     : "#1D9E75",
    "Caution"  : "#EF9F27",
    "Critical" : "#D85A30"
}

batch = agg["batch_size"]

# ── PLOT 1 — PRI Score bar chart ──
ax1 = axes[0]

bar_colors = [risk_colors[r] for r in agg["risk_level"]]
bars = ax1.bar(batch, agg["PRI_score"],
               color=bar_colors, alpha=0.85, width=20)

# Threshold lines
ax1.axhline(y=70, color="#D85A30", linewidth=2,
            linestyle="--", label="Critical threshold (70)")
ax1.axhline(y=40, color="#EF9F27", linewidth=2,
            linestyle=":", label="Caution threshold (40)")

# Annotate cliff point
ax1.annotate(
    f"Performance cliff\nbatch={cliff_batch}\nPRI={cliff_score}",
    xy=(cliff_batch, cliff_score),
    xytext=(cliff_batch - 150, cliff_score - 15),
    fontsize=9, color="#D85A30",
    arrowprops=dict(arrowstyle="->",
                    color="#D85A30", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="white", edgecolor="#D85A30")
)

# Add score labels on bars
for bar, score in zip(bars, agg["PRI_score"]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{score}",
        ha="center", va="bottom",
        fontsize=7, fontweight="bold"
    )

ax1.set_title(
    "Performance Risk Index (PRI) — Combined Risk Score Per Batch Size\n"
    "(higher = more dangerous to operate at this batch size)",
    fontsize=11, fontweight="bold"
)
ax1.set_ylabel("PRI Score (0-100)")
ax1.set_xticks(batch)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_ylim(0, 115)

# ── PLOT 2 — 4 signal breakdown ──
ax2 = axes[1]

ax2.plot(batch, agg["sla_proximity"] * 100,
         color="#D85A30", linewidth=2,
         marker="o", markersize=5,
         label="SLA proximity (weight 35%)")
ax2.plot(batch, agg["latency_growth_rate"] * 100,
         color="#534AB7", linewidth=2,
         marker="s", markersize=5,
         label="Latency growth rate (weight 30%)")
ax2.plot(batch, agg["tail_risk"] * 100,
         color="#EF9F27", linewidth=2,
         marker="^", markersize=5,
         label="Tail risk (weight 20%)")
ax2.plot(batch, agg["throughput_collapse"] * 100,
         color="#1D9E75", linewidth=2,
         marker="D", markersize=5,
         label="Throughput collapse (weight 15%)")

ax2.set_title(
    "PRI Signal Breakdown — Individual Risk Components\n"
    "(shows which signal is driving the overall risk score)",
    fontsize=11, fontweight="bold"
)
ax2.set_ylabel("Signal Score (0-100)")
ax2.set_xticks(batch)
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(True, alpha=0.3)

# ── PLOT 3 — Latency with risk shading ──
ax3 = axes[2]

for _, row in agg.iterrows():
    ax3.axvspan(
        row["batch_size"] - 8,
        row["batch_size"] + 8,
        alpha=0.15,
        color=risk_colors[row["risk_level"]]
    )

ax3.plot(batch, agg["avg_latency_ms"],
         color="#1D9E75", linewidth=2.5,
         marker="o", markersize=6,
         label="Avg latency")
ax3.plot(batch, agg["p95_latency_ms"],
         color="#EF9F27", linewidth=2,
         marker="s", markersize=5,
         linestyle="--", label="p95 latency")
ax3.plot(batch, agg["p99_latency_ms"],
         color="#534AB7", linewidth=1.5,
         marker="^", markersize=4,
         linestyle=":", label="p99 latency")
ax3.axhline(y=SLA_THRESHOLD_MS, color="red",
            linewidth=1.5, linestyle="-.",
            label="SLA threshold (20ms)")

# Cliff annotation
ax3.axvline(x=cliff_batch, color="#D85A30",
            linewidth=2, linestyle="--", alpha=0.7)
ax3.annotate(
    f"Cliff point\nbatch={cliff_batch}",
    xy=(cliff_batch, cliff_lat),
    xytext=(cliff_batch + 40, cliff_lat + 2),
    fontsize=9, color="#D85A30",
    arrowprops=dict(arrowstyle="->",
                    color="#D85A30", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="white",
              edgecolor="#D85A30")
)

ax3.set_title(
    "Latency Curve — Risk-Shaded Operating Zones",
    fontsize=11, fontweight="bold"
)
ax3.set_ylabel("Latency (ms)")
ax3.set_xlabel("Batch size")
ax3.set_xticks(batch)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ── RISK LEGEND ──
legend_elements = [
    Patch(facecolor="#1D9E75", alpha=0.5,
          label="Safe zone (PRI < 40)"),
    Patch(facecolor="#EF9F27", alpha=0.5,
          label="Caution zone (PRI 40-70)"),
    Patch(facecolor="#D85A30", alpha=0.5,
          label="Critical zone (PRI > 70)"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center", ncol=3,
    fontsize=10, bbox_to_anchor=(0.5, 0.01)
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    "/Users/devansh/Desktop/optimizeation/day11_risk_dashboard.png",
    dpi=150, bbox_inches="tight"
)
print("✅ Saved → day11_risk_dashboard.png")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("\n── Day 11 Findings ──\n")
print(f"Performance cliff identified at : batch={cliff_batch}")
print(f"Cliff PRI score                 : {cliff_score}/100")
print(f"Cliff avg latency               : {cliff_lat:.2f}ms")

print(f"\nFull Risk Report:\n")
print(agg[[
    "batch_size", "avg_latency_ms", "p95_latency_ms",
    "PRI_score", "risk_level"
]].to_string(index=False))

print("\n── Risk Zone Summary ──\n")
for level in ["Safe", "Caution", "Critical"]:
    batches = agg[agg["risk_level"] == level]["batch_size"].tolist()
    print(f"  {level:<10} → {batches}")

print("\n── PRI Signal Weights ──\n")
print("  SLA proximity      35% — how close p95 is to 20ms limit")
print("  Latency growth     30% — how much latency grew vs minimum")
print("  Tail risk          20% — gap between p99 and avg latency")
print("  Throughput drop    15% — how much throughput fell vs peak")