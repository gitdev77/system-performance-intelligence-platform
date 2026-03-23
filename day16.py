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

df["cpu_usage"]      = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage"]      = df["cpu_usage"].ffill()
df["avg_latency_ms"] = df["avg_latency"] * 1000
df["p95_latency_ms"] = df["p95_latency"] * 1000

# ─────────────────────────────────────────
# AGGREGATE by batch size
# ─────────────────────────────────────────
agg = df.groupby("batch_size").agg(
    avg_latency_ms = ("avg_latency_ms",  "mean"),
    p95_latency_ms = ("p95_latency_ms",  "mean"),
    cpu_usage      = ("cpu_usage",       "mean"),
    memory_usage   = ("memory_usage",    "mean"),
    throughput     = ("throughput",      "mean"),
    cv_latency     = ("cv_latency",      "mean"),
).reset_index()

# ─────────────────────────────────────────
# STEP 1 — Normalize all resources to
# 0-100 scale so we can compare them
# fairly on the same chart
#
# Normalization formula:
# (value - min) / (max - min) * 100
#
# 0  = resource at its most relaxed point
# 100 = resource at its most stressed point
# ─────────────────────────────────────────
def normalize(series):
    mn = series.min()
    mx = series.max()
    if mx == mn:
        return pd.Series([50.0] * len(series))
    return (series - mn) / (mx - mn) * 100

agg["cpu_norm"]        = normalize(agg["cpu_usage"])
agg["memory_norm"]     = normalize(agg["memory_usage"])
agg["throughput_norm"] = normalize(
    agg["throughput"].max() - agg["throughput"]
)  # invert throughput — lower throughput = higher stress
agg["latency_norm"]    = normalize(agg["avg_latency_ms"])
agg["cv_norm"]         = normalize(agg["cv_latency"])

# ─────────────────────────────────────────
# STEP 2 — Identify dominant resource
# per batch size
# Dominant = whichever resource has
# highest normalized stress score
# at that batch size
# ─────────────────────────────────────────
resource_cols = {
    "throughput_norm" : "Throughput Collapse",
    "latency_norm"    : "Latency Growth",
    "cv_norm"         : "Latency Inconsistency",
    "cpu_norm"        : "CPU Pressure",
    "memory_norm"     : "Memory Pressure",
}

agg["dominant_resource"] = agg[
    list(resource_cols.keys())
].idxmax(axis=1).map(resource_cols)

# ─────────────────────────────────────────
# STEP 3 — Find transition points
# Where dominant resource changes
# = system bottleneck shifts
# ─────────────────────────────────────────
agg["prev_dominant"] = agg["dominant_resource"].shift(1)
agg["transition"]    = (
    agg["dominant_resource"] != agg["prev_dominant"]
).astype(int)
agg.loc[agg.index[0], "transition"] = 0

transitions = agg[agg["transition"] == 1][
    ["batch_size", "dominant_resource", "prev_dominant"]
]

# ─────────────────────────────────────────
# STEP 4 — Resource saturation score
# Combined stress level at each batch size
# Average of all normalized resources
# ─────────────────────────────────────────
agg["saturation_score"] = agg[
    list(resource_cols.keys())
].mean(axis=1).round(1)

# ─────────────────────────────────────────
# PLOT — 4 panel resource dashboard
# ─────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(13, 18))
fig.suptitle(
    "Resource Saturation Profiling — Dominant Limiting Resource\n"
    "System Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

resource_colors = {
    "Throughput Collapse"    : "#1D9E75",
    "Latency Growth"         : "#534AB7",
    "Latency Inconsistency"  : "#EF9F27",
    "CPU Pressure"           : "#D85A30",
    "Memory Pressure"        : "#0F6E56",
}

batch = agg["batch_size"]

# ── PLOT 1 — Normalized resource lines ──
ax1 = axes[0]

for col, label in resource_cols.items():
    ax1.plot(
        batch, agg[col],
        color=resource_colors[label],
        linewidth=2, marker="o",
        markersize=4, label=label
    )

# Mark transition points
for _, row in transitions.iterrows():
    ax1.axvline(
        x=row["batch_size"],
        color="gray", linewidth=1,
        linestyle="--", alpha=0.5
    )
    ax1.text(
        row["batch_size"],
        95,
        f"shift\n@{int(row['batch_size'])}",
        ha="center", fontsize=7,
        color="gray"
    )

ax1.set_title(
    "Normalized Resource Stress Levels (0-100)\n"
    "(higher = more stressed — vertical lines = bottleneck shifts)",
    fontsize=11, fontweight="bold"
)
ax1.set_ylabel("Stress Level (normalized 0-100)")
ax1.set_xticks(batch)
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-5, 110)

# ── PLOT 2 — Dominant resource per batch ──
ax2 = axes[1]

dom_colors = [
    resource_colors[r] for r in agg["dominant_resource"]
]
bars = ax2.bar(
    batch,
    [100] * len(batch),
    color=dom_colors,
    alpha=0.85, width=20
)

# Labels
for bar, dom in zip(bars, agg["dominant_resource"]):
    short_name = dom.split()[0]  # first word only
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        50,
        short_name,
        ha="center", va="center",
        fontsize=8, fontweight="bold",
        color="white", rotation=90
    )

ax2.set_title(
    "Dominant Limiting Resource Per Batch Size\n"
    "(which resource is the bottleneck at each operating point)",
    fontsize=11, fontweight="bold"
)
ax2.set_ylabel("Dominant Resource")
ax2.set_xticks(batch)
ax2.set_yticks([])
ax2.grid(False)

# ── PLOT 3 — Overall saturation score ──
ax3 = axes[2]

sat_colors = [
    "#D85A30" if s > 66 else
    "#EF9F27" if s > 33 else
    "#1D9E75"
    for s in agg["saturation_score"]
]
bars3 = ax3.bar(
    batch,
    agg["saturation_score"],
    color=sat_colors,
    alpha=0.85, width=20
)

ax3.axhline(y=66, color="#D85A30", linewidth=2,
            linestyle="--", label="High saturation (66)")
ax3.axhline(y=33, color="#EF9F27", linewidth=1.5,
            linestyle=":", label="Moderate saturation (33)")

for bar, score in zip(bars3, agg["saturation_score"]):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{score}",
        ha="center", va="bottom",
        fontsize=8, fontweight="bold"
    )

ax3.set_title(
    "Overall System Saturation Score Per Batch Size\n"
    "(average stress across all resources — higher = more dangerous)",
    fontsize=11, fontweight="bold"
)
ax3.set_ylabel("Saturation Score (0-100)")
ax3.set_xticks(batch)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis="y")

# ── PLOT 4 — Resource competition heatmap ──
ax4 = axes[3]

heatmap_data = agg[list(resource_cols.keys())].T.values
im = ax4.imshow(
    heatmap_data,
    cmap="RdYlGn_r",
    aspect="auto",
    vmin=0, vmax=100
)

ax4.set_xticks(range(len(batch)))
ax4.set_xticklabels(batch, rotation=45, fontsize=8)
ax4.set_yticks(range(len(resource_cols)))
ax4.set_yticklabels(
    list(resource_cols.values()),
    fontsize=8
)

# Annotate heatmap cells
for i in range(len(resource_cols)):
    for j in range(len(batch)):
        val = heatmap_data[i, j]
        ax4.text(
            j, i, f"{val:.0f}",
            ha="center", va="center",
            fontsize=7, fontweight="bold",
            color="white" if val > 50 else "black"
        )

plt.colorbar(im, ax=ax4, shrink=0.8,
             label="Stress Level (0-100)")
ax4.set_title(
    "Resource Competition Heatmap\n"
    "(red = highly stressed, green = relaxed)",
    fontsize=11, fontweight="bold"
)

# ── LEGEND ──
legend_elements = [
    Patch(facecolor=c, alpha=0.8, label=l)
    for l, c in resource_colors.items()
]
fig.legend(
    handles=legend_elements,
    loc="lower center", ncol=5,
    fontsize=8, bbox_to_anchor=(0.5, 0.01)
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    "/Users/devansh/Desktop/optimizeation/day16_saturation.png",
    dpi=150, bbox_inches="tight"
)
print("✅ Saved → day16_saturation.png")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("\n── Day 16 Findings ──\n")
print(agg[[
    "batch_size", "dominant_resource",
    "saturation_score", "transition"
]].to_string(index=False))

print("\n── Bottleneck Transition Points ──\n")
if len(transitions) > 0:
    for _, row in transitions.iterrows():
        print(
            f"  batch={int(row['batch_size']):<6} "
            f"bottleneck shifts from "
            f"'{row['prev_dominant']}' "
            f"→ '{row['dominant_resource']}'"
        )
else:
    print("  No transitions detected — "
          "same dominant resource throughout")

print("\n── Resource Summary ──\n")
dom_counts = agg["dominant_resource"].value_counts()
for resource, count in dom_counts.items():
    batches = agg[
        agg["dominant_resource"] == resource
    ]["batch_size"].tolist()
    print(f"  {resource:<25} "
          f"dominates at {count} batch sizes → {batches}")