import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("/Users/devansh/Desktop/optimizeation/telemetry_v2.csv")

# Fix CPU psutil bug (0.0% readings on Apple Silicon)
df["cpu_usage"] = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage"] = df["cpu_usage"].ffill()

# Convert latency to ms (milliseconds — thousandths of a second)
df["avg_latency_ms"] = df["avg_latency"] * 1000

# ─────────────────────────────────────────
# AGGREGATE by batch size
# (mean of all 8 repeats per batch size)
# ─────────────────────────────────────────
agg = df.groupby("batch_size").agg(
    avg_latency_ms = ("avg_latency_ms", "mean"),
    cpu_usage      = ("cpu_usage",      "mean"),
    p95_latency_ms = ("p95_latency",    lambda x: x.mean() * 1000),
    throughput     = ("throughput",     "mean"),
).reset_index()

# ─────────────────────────────────────────
# ROLLING AVERAGE (smoothing — averages
# nearby points to remove noise/spikes)
# window=3 means average of 3 consecutive
# batch sizes at a time
# ─────────────────────────────────────────
agg["latency_smooth"]  = agg["avg_latency_ms"].rolling(window=3, center=True).mean()
agg["cpu_smooth"]      = agg["cpu_usage"].rolling(window=3, center=True).mean()

# ─────────────────────────────────────────
# FIND CROSSOVER POINT
# Where CPU growth rate starts matching
# latency growth rate
# ─────────────────────────────────────────
agg["latency_growth"] = agg["avg_latency_ms"].pct_change()  # % change between steps
agg["cpu_growth"]     = agg["cpu_usage"].pct_change()

# Crossover = first point where both are
# growing at more than 5% simultaneously
crossover = agg[
    (agg["latency_growth"] > 0.05) &
    (agg["cpu_growth"] > 0.05)
]
crossover_batch = crossover["batch_size"].min() if len(crossover) > 0 else None

# ─────────────────────────────────────────
# PLOT — Dual axis chart
# (two y-axes on same chart — left for
# latency, right for CPU)
# ─────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(14, 7))

fig.suptitle(
    "Latency vs CPU Usage — Dual Axis Analysis\nSystem Performance Intelligence Platform",
    fontsize=14, fontweight="bold"
)

# ── LEFT AXIS — Latency ──
color_latency = "#1D9E75"
color_p95     = "#EF9F27"

ax1.set_xlabel("Batch size (number of samples processed at once)", fontsize=11)
ax1.set_ylabel("Latency (ms)", color=color_latency, fontsize=11)

# Raw latency dots
ax1.scatter(df["batch_size"], df["avg_latency_ms"],
            alpha=0.25, color=color_latency, s=20, label="_nolegend_")

# Aggregated line
ax1.plot(agg["batch_size"], agg["avg_latency_ms"],
         color=color_latency, linewidth=2, marker="o",
         markersize=6, label="Avg latency (mean per batch)")

# Smoothed line
ax1.plot(agg["batch_size"], agg["latency_smooth"],
         color=color_latency, linewidth=3, linestyle="--",
         alpha=0.6, label="Latency rolling average (smoothed)")

# p95 line
ax1.plot(agg["batch_size"], agg["p95_latency_ms"],
         color=color_p95, linewidth=2, marker="s",
         markersize=5, linestyle=":", label="p95 latency")

# SLA threshold line
ax1.axhline(y=20, color="red", linewidth=1.5,
            linestyle="-.", alpha=0.7, label="SLA threshold (20ms)")

ax1.tick_params(axis="y", labelcolor=color_latency)
ax1.set_xticks(agg["batch_size"])
ax1.grid(True, alpha=0.2)

# ── RIGHT AXIS — CPU ──
ax2 = ax1.twinx()  # twinx = share same x-axis, separate y-axis on right
color_cpu = "#534AB7"

ax2.set_ylabel("CPU Usage (%)", color=color_cpu, fontsize=11)

# Raw CPU dots
ax2.scatter(df["batch_size"], df["cpu_usage"],
            alpha=0.15, color=color_cpu, s=20, label="_nolegend_")

# Aggregated CPU line
ax2.plot(agg["batch_size"], agg["cpu_usage"],
         color=color_cpu, linewidth=2, marker="^",
         markersize=6, linestyle="-", label="CPU usage (mean per batch)")

# Smoothed CPU line
ax2.plot(agg["batch_size"], agg["cpu_smooth"],
         color=color_cpu, linewidth=3, linestyle="--",
         alpha=0.6, label="CPU rolling average (smoothed)")

ax2.tick_params(axis="y", labelcolor=color_cpu)
ax2.set_ylim(0, 120)

# ── ANNOTATIONS ──
# Inflection point (batch=192 from Day 7)
inflection_lat = agg[agg["batch_size"] == 192]["avg_latency_ms"].values[0]
ax1.annotate(
    "Inflection point\n(batch=192)",
    xy=(192, inflection_lat),
    xytext=(220, inflection_lat + 4),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
)

# Crossover point annotation
if crossover_batch and not pd.isna(crossover_batch):
    cross_lat = agg[agg["batch_size"] == crossover_batch]["avg_latency_ms"].values[0]
    ax1.annotate(
        f"CPU+Latency\nboth growing\nbatch={crossover_batch}",
        xy=(crossover_batch, cross_lat),
        xytext=(crossover_batch + 60, cross_lat + 6),
        fontsize=9,
        color="#534AB7",
        arrowprops=dict(arrowstyle="->", color="#534AB7", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#534AB7")
    )

# ── LEGEND ──
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig("/Users/devansh/Desktop/optimizeation/day9_latency_cpu.png",
            dpi=150, bbox_inches="tight")
print("✅ Saved → day9_latency_cpu.png")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("\n── Day 9 Findings ──\n")
print(agg[[
    "batch_size", "avg_latency_ms",
    "cpu_usage", "latency_growth", "cpu_growth"
]].round(3).to_string(index=False))

if crossover_batch and not pd.isna(crossover_batch):
    print(f"\n⚡ CPU + Latency both growing simultaneously from batch={crossover_batch}")
else:
    print("\n⚡ No clear simultaneous crossover detected — CPU and latency grow independently")