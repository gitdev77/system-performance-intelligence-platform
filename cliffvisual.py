import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
agg = pd.read_csv("cliff_analysis.csv")

# Convert to milliseconds for readability
agg["avg_latency_ms"] = agg["avg_latency"] * 1000
agg["p95_latency_ms"] = agg["p95_latency"] * 1000
agg["p99_latency_ms"] = agg["p99_latency"] * 1000

SLA_THRESHOLD_MS = 20.0

# ─────────────────────────────────────────
# FIGURE SETUP — 3 subplots
# ─────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 14))
fig.suptitle(
    "Performance Cliff Detection Report\nSystem Performance Intelligence Platform",
    fontsize=15, fontweight="bold", y=0.98
)

batch = agg["batch_size"]
colors = {
    "healthy"    : "#1D9E75",
    "stress"     : "#EF9F27",
    "cliff"      : "#D85A30",
    "sla"        : "#E24B4A",
    "derivative" : "#534AB7",
    "p99"        : "#0F6E56",
}

# ─────────────────────────────────────────
# PLOT 1 — Latency curve with cliff zones
# ─────────────────────────────────────────
ax1 = axes[0]

# Shade saturation zone
sat_start = agg[agg["saturation_zone"] == 1]["batch_size"].min()
ax1.axvspan(sat_start, batch.max(), alpha=0.08, color=colors["cliff"],
            label="Saturation zone")

# Shade healthy zone
ax1.axvspan(batch.min(), sat_start, alpha=0.06, color=colors["healthy"],
            label="Healthy zone")

# Plot latency lines
ax1.plot(batch, agg["avg_latency_ms"], color=colors["healthy"],
         linewidth=2.5, marker="o", markersize=5, label="Avg latency")
ax1.plot(batch, agg["p95_latency_ms"], color=colors["stress"],
         linewidth=2, marker="s", markersize=4, linestyle="--", label="p95 latency")
ax1.plot(batch, agg["p99_latency_ms"], color=colors["p99"],
         linewidth=1.5, marker="^", markersize=4, linestyle=":", label="p99 latency")

# SLA threshold line
ax1.axhline(y=SLA_THRESHOLD_MS, color=colors["sla"], linewidth=1.5,
            linestyle="-.", label=f"SLA threshold ({SLA_THRESHOLD_MS}ms)")

# Annotate inflection point
inflection_batch = 192
inflection_lat = agg[agg["batch_size"] == inflection_batch]["avg_latency_ms"].values[0]
ax1.annotate(
    f"Inflection point\nbatch={inflection_batch}",
    xy=(inflection_batch, inflection_lat),
    xytext=(inflection_batch + 40, inflection_lat + 3),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
)

# Annotate SLA breach
sla_breach_batch = 512
sla_breach_p95 = agg[agg["batch_size"] == sla_breach_batch]["p95_latency_ms"].values[0]
ax1.annotate(
    f"SLA breach\np95={sla_breach_p95:.1f}ms",
    xy=(sla_breach_batch, sla_breach_p95),
    xytext=(sla_breach_batch - 120, sla_breach_p95 + 2),
    fontsize=9,
    color=colors["sla"],
    arrowprops=dict(arrowstyle="->", color=colors["sla"], lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=colors["sla"])
)

ax1.set_title("Latency Degradation Curve — Avg, p95, p99", fontsize=11,
              fontweight="bold")
ax1.set_xlabel("Batch size")
ax1.set_ylabel("Latency (ms)")
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(batch)

# ─────────────────────────────────────────
# PLOT 2 — Latency derivative (cliff signal)
# ─────────────────────────────────────────
ax2 = axes[1]

deriv = agg["latency_derivative"] * 1000  # convert to ms per unit
cliff_threshold = deriv.mean() + 1.5 * deriv.std()

# Color bars by cliff flag
bar_colors = [colors["cliff"] if f == 1 else colors["derivative"]
              for f in agg["cliff_flag"]]

ax2.bar(batch, deriv, color=bar_colors, alpha=0.8, width=20)
ax2.axhline(y=cliff_threshold, color=colors["sla"], linewidth=1.5,
            linestyle="-.", label=f"Cliff threshold")

# Annotate false positive
ax2.annotate(
    "False positive\n(measurement noise)",
    xy=(16, deriv[agg["batch_size"] == 16].values[0]),
    xytext=(80, deriv.max() * 0.85),
    fontsize=8,
    arrowprops=dict(arrowstyle="->", color="gray", lw=1),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
)

ax2.set_title("Latency Derivative — Rate of Change (Cliff Signal)", fontsize=11,
              fontweight="bold")
ax2.set_xlabel("Batch size")
ax2.set_ylabel("dLatency/dBatch (ms)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_xticks(batch)

# ─────────────────────────────────────────
# PLOT 3 — Throughput degradation
# ─────────────────────────────────────────
ax3 = axes[2]

ax3.fill_between(batch, agg["throughput"], alpha=0.15, color=colors["healthy"])
ax3.plot(batch, agg["throughput"], color=colors["healthy"],
         linewidth=2.5, marker="o", markersize=5, label="Throughput (req/s)")

# Shade saturation zone
ax3.axvspan(sat_start, batch.max(), alpha=0.08, color=colors["cliff"])

# Annotate peak throughput
peak_idx = agg["throughput"].idxmax()
peak_batch = agg.loc[peak_idx, "batch_size"]
peak_tput = agg.loc[peak_idx, "throughput"]
ax3.annotate(
    f"Peak throughput\n{peak_tput:.0f} req/s @ batch={peak_batch}",
    xy=(peak_batch, peak_tput),
    xytext=(peak_batch + 60, peak_tput - 20),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray")
)

ax3.set_title("Throughput Degradation — Requests per Second", fontsize=11,
              fontweight="bold")
ax3.set_xlabel("Batch size")
ax3.set_ylabel("Throughput (req/s)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(batch)

# ─────────────────────────────────────────
# FINAL TOUCHES
# ─────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("cliff_detection_report.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved → cliff_detection_report.png")