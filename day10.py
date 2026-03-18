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

# ─────────────────────────────────────────
# AGGREGATE by batch size
# ─────────────────────────────────────────
agg = df.groupby("batch_size").agg(
    avg_latency_ms = ("avg_latency_ms", "mean"),
    p95_latency_ms = ("p95_latency_ms", "mean"),
    cpu_usage      = ("cpu_usage",      "mean"),
    throughput     = ("throughput",     "mean"),
).reset_index()

# ─────────────────────────────────────────
# STEP 1 — First derivative
# (how fast latency grows per batch step)
# ─────────────────────────────────────────
agg["d1"] = np.gradient(
    agg["avg_latency_ms"],
    agg["batch_size"]
)

# ─────────────────────────────────────────
# STEP 2 — Second derivative
# (how fast the growth is accelerating)
# High d2 = latency accelerating = danger
# ─────────────────────────────────────────
agg["d2"] = np.gradient(
    agg["d1"],
    agg["batch_size"]
)

# ─────────────────────────────────────────
# STEP 3 — Calculate threshold
# mean + 0.5 std = more sensitive detection
# ─────────────────────────────────────────
d2_mean = agg["d2"].mean()
d2_std  = agg["d2"].std()
nonlinear_threshold = d2_mean + 0.5 * d2_std

# ─────────────────────────────────────────
# STEP 4 — Classify zones
# ─────────────────────────────────────────
def classify_zone(row):
    if row["d2"] > nonlinear_threshold:
        return "Saturated"
    elif row["d2"] > d2_mean:
        return "Warning"
    else:
        return "Healthy"

agg["zone"] = agg.apply(classify_zone, axis=1)

# ─────────────────────────────────────────
# STEP 5b — Require 2 consecutive Saturated
# zones to confirm real saturation
# (filters out single-point noise spikes)
# ─────────────────────────────────────────
agg["prev_zone"] = agg["zone"].shift(1)  # shift(1) = look at previous row
agg["zone"] = agg.apply(
    lambda row: row["zone"]
    if (row["zone"] == "Saturated" and row["prev_zone"] == "Saturated")
    else ("Warning" if row["zone"] == "Saturated" else row["zone"]),
    axis=1
)
agg.drop(columns=["prev_zone"], inplace=True)

# ─────────────────────────────────────────
# STEP 5 — Fix boundary effect
# First 2 rows always have unreliable d2
# due to edge math — force them to Healthy
# ─────────────────────────────────────────
agg.loc[agg.index[:2], "zone"] = "Healthy"

# ─────────────────────────────────────────
# STEP 6 — Find saturation threshold
# First batch size in Saturated zone
# after boundary fix
# ─────────────────────────────────────────
saturated = agg[agg["zone"] == "Saturated"]
saturation_threshold = int(saturated["batch_size"].min()) \
    if len(saturated) > 0 else None

# ─────────────────────────────────────────
# PLOT — 3 panel chart
# ─────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 14))
fig.suptitle(
    "Non-Linear Growth Detection Engine\nSystem Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

colors = {
    "Healthy"   : "#1D9E75",
    "Warning"   : "#EF9F27",
    "Saturated" : "#D85A30"
}

batch = agg["batch_size"]

# ── PLOT 1 — Latency curve with zone shading ──
ax1 = axes[0]

for _, row in agg.iterrows():
    ax1.axvspan(
        row["batch_size"] - 8,
        row["batch_size"] + 8,
        alpha=0.15,
        color=colors[row["zone"]]
    )

ax1.plot(batch, agg["avg_latency_ms"],
         color="#1D9E75", linewidth=2.5,
         marker="o", markersize=6,
         label="Avg latency")
ax1.plot(batch, agg["p95_latency_ms"],
         color="#EF9F27", linewidth=2,
         marker="s", markersize=5,
         linestyle="--", label="p95 latency")
ax1.axhline(y=20, color="red", linewidth=1.5,
            linestyle="-.", label="SLA threshold (20ms)")

if saturation_threshold:
    sat_lat = agg[
        agg["batch_size"] == saturation_threshold
    ]["avg_latency_ms"].values[0]
    ax1.axvline(x=saturation_threshold,
                color="#D85A30", linewidth=2,
                linestyle="--", alpha=0.7)
    ax1.annotate(
        f"Saturation threshold\nbatch={saturation_threshold}",
        xy=(saturation_threshold, sat_lat),
        xytext=(saturation_threshold + 40, sat_lat + 3),
        fontsize=9, color="#D85A30",
        arrowprops=dict(arrowstyle="->",
                        color="#D85A30", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="white",
                  edgecolor="#D85A30")
    )
else:
    ax1.text(
        0.5, 0.92,
        "No saturation detected — adjusting threshold",
        transform=ax1.transAxes, fontsize=9,
        ha="center", color="#D85A30",
        bbox=dict(boxstyle="round",
                  facecolor="white",
                  edgecolor="#D85A30")
    )

ax1.set_title("Latency Curve — Zone Classification",
              fontsize=11, fontweight="bold")
ax1.set_ylabel("Latency (ms)")
ax1.set_xticks(batch)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# ── PLOT 2 — First derivative ──
ax2 = axes[1]

bar_colors2 = [colors[z] for z in agg["zone"]]
ax2.bar(batch, agg["d1"],
        color=bar_colors2, alpha=0.8, width=20)
ax2.axhline(y=0, color="black", linewidth=0.8)
ax2.set_title(
    "First Derivative — Rate of Latency Change\n"
    "(how fast latency grows per unit batch size increase)",
    fontsize=11, fontweight="bold"
)
ax2.set_ylabel("dLatency/dBatch")
ax2.set_xticks(batch)
ax2.grid(True, alpha=0.3, axis="y")

# ── PLOT 3 — Second derivative ──
ax3 = axes[2]

bar_colors3 = [colors[z] for z in agg["zone"]]
ax3.bar(batch, agg["d2"],
        color=bar_colors3, alpha=0.8, width=20)
ax3.axhline(y=nonlinear_threshold, color="#D85A30",
            linewidth=2, linestyle="--",
            label="Non-linear threshold (mean + 0.5 std)")
ax3.axhline(y=d2_mean, color="#EF9F27",
            linewidth=1.5, linestyle=":",
            label="Mean d2")
ax3.axhline(y=0, color="black", linewidth=0.8)

ax3.set_title(
    "Second Derivative — Acceleration of Latency Growth\n"
    "(bars above red line = non-linear danger zone)",
    fontsize=11, fontweight="bold"
)
ax3.set_ylabel("d²Latency/dBatch²")
ax3.set_xlabel("Batch size")
ax3.set_xticks(batch)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis="y")

# ── ZONE LEGEND ──
legend_elements = [
    Patch(facecolor="#1D9E75", alpha=0.4, label="Healthy zone"),
    Patch(facecolor="#EF9F27", alpha=0.4, label="Warning zone"),
    Patch(facecolor="#D85A30", alpha=0.4, label="Saturated zone"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center", ncol=3,
    fontsize=10, bbox_to_anchor=(0.5, 0.01)
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    "/Users/devansh/Desktop/optimizeation/day10_nonlinear.png",
    dpi=150, bbox_inches="tight"
)
print("✅ Saved → day10_nonlinear.png")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("\n── Day 10 Findings ──\n")
print(f"d2 mean                : {d2_mean:.6f}")
print(f"d2 std                 : {d2_std:.6f}")
print(f"Non-linear threshold   : {nonlinear_threshold:.6f}")
print(f"Saturation threshold   : batch={saturation_threshold}")

print(f"\nZone classification per batch size:\n")
print(agg[[
    "batch_size", "avg_latency_ms",
    "d1", "d2", "zone"
]].round(6).to_string(index=False))

print("\n── Zone Summary ──\n")
zone_counts = agg["zone"].value_counts()
for zone, count in zone_counts.items():
    batches = agg[agg["zone"] == zone]["batch_size"].tolist()
    print(f"  {zone:<12} {count} batch sizes → {batches}")