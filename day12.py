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

SLA_THRESHOLD_MS = 20.0

# ─────────────────────────────────────────
# STEP 1 — Recalculate SLA breach
# with correct 20ms threshold
# (original used 150ms — too generous)
# sla_breach_v2 = 1 if p95 > 20ms else 0
# ─────────────────────────────────────────
df["sla_breach_v2"] = (
    df["p95_latency_ms"] > SLA_THRESHOLD_MS
).astype(int)

# ─────────────────────────────────────────
# STEP 2 — Breach rate per batch size
# breach rate = how often (% of runs)
# p95 crossed 20ms at that batch size
# ─────────────────────────────────────────
breach_stats = df.groupby("batch_size").agg(
    total_runs    = ("sla_breach_v2", "count"),
    breach_count  = ("sla_breach_v2", "sum"),
    breach_rate   = ("sla_breach_v2", "mean"),
    avg_p95_ms    = ("p95_latency_ms", "mean"),
    max_p95_ms    = ("p95_latency_ms", "max"),
    avg_latency_ms= ("avg_latency_ms", "mean"),
).reset_index()

# Convert breach rate to percentage
breach_stats["breach_rate_pct"] = (
    breach_stats["breach_rate"] * 100
).round(1)

# ─────────────────────────────────────────
# STEP 3 — Breach severity
# How far over 20ms did p95 go on average?
# severity = avg_p95 - SLA_threshold
# Negative = under threshold (good)
# Positive = over threshold (bad)
# ─────────────────────────────────────────
breach_stats["breach_severity_ms"] = (
    breach_stats["avg_p95_ms"] - SLA_THRESHOLD_MS
).round(2)

# ─────────────────────────────────────────
# STEP 4 — Error budget consumption
# If SLA allows 0.1% breach rate,
# how much of that budget are we using?
# budget_consumed = breach_rate / 0.001
# 100% = fully consumed budget
# ─────────────────────────────────────────
SLA_ALLOWED_BREACH_RATE = 0.001  # 0.1% allowed
breach_stats["budget_consumed_pct"] = (
    breach_stats["breach_rate"] /
    SLA_ALLOWED_BREACH_RATE * 100
).clip(0, 1000).round(1)

# ─────────────────────────────────────────
# STEP 5 — Classify breach severity
# Safe     → breach rate < 10%
# Warning  → breach rate 10-50%
# Critical → breach rate > 50%
# ─────────────────────────────────────────
def classify_breach(rate):
    if rate >= 50:
        return "Critical"
    elif rate >= 10:
        return "Warning"
    else:
        return "Safe"

breach_stats["breach_class"] = (
    breach_stats["breach_rate_pct"].apply(classify_breach)
)

# ─────────────────────────────────────────
# PLOT — 4 panel breach dashboard
# ─────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(13, 18))
fig.suptitle(
    "SLA Breach Frequency Analysis\nSystem Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

risk_colors = {
    "Safe"     : "#1D9E75",
    "Warning"  : "#EF9F27",
    "Critical" : "#D85A30"
}

batch = breach_stats["batch_size"]

# ── PLOT 1 — Breach rate per batch size ──
ax1 = axes[0]

bar_colors = [
    risk_colors[c] for c in breach_stats["breach_class"]
]
bars = ax1.bar(batch, breach_stats["breach_rate_pct"],
               color=bar_colors, alpha=0.85, width=20)

# Threshold lines
ax1.axhline(y=50, color="#D85A30", linewidth=2,
            linestyle="--", label="Critical threshold (50%)")
ax1.axhline(y=10, color="#EF9F27", linewidth=2,
            linestyle=":", label="Warning threshold (10%)")

# Score labels on bars
for bar, rate in zip(bars, breach_stats["breach_rate_pct"]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{rate}%",
        ha="center", va="bottom",
        fontsize=8, fontweight="bold"
    )

ax1.set_title(
    "SLA Breach Rate per Batch Size\n"
    "(% of runs where p95 latency exceeded 20ms)",
    fontsize=11, fontweight="bold"
)
ax1.set_ylabel("Breach Rate (%)")
ax1.set_xticks(batch)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_ylim(0, 110)

# ── PLOT 2 — Breach severity ──
ax2 = axes[1]

severity_colors = [
    "#D85A30" if s > 0 else "#1D9E75"
    for s in breach_stats["breach_severity_ms"]
]
bars2 = ax2.bar(batch,
                breach_stats["breach_severity_ms"],
                color=severity_colors,
                alpha=0.85, width=20)

ax2.axhline(y=0, color="red", linewidth=2,
            linestyle="-.", label="SLA boundary (20ms)")

# Labels
for bar, sev in zip(bars2, breach_stats["breach_severity_ms"]):
    ypos = bar.get_height() + 0.1 if bar.get_height() >= 0 \
        else bar.get_height() - 0.5
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        ypos,
        f"{sev:+.1f}ms",
        ha="center", va="bottom",
        fontsize=7, fontweight="bold"
    )

ax2.set_title(
    "Breach Severity — How Far Over 20ms Did p95 Go?\n"
    "(positive = over SLA, negative = under SLA — good)",
    fontsize=11, fontweight="bold"
)
ax2.set_ylabel("Excess Over SLA (ms)")
ax2.set_xticks(batch)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis="y")

# ── PLOT 3 — Error budget consumption ──
ax3 = axes[2]

budget_colors = [
    "#D85A30" if b > 100 else
    "#EF9F27" if b > 50 else
    "#1D9E75"
    for b in breach_stats["budget_consumed_pct"]
]
bars3 = ax3.bar(batch,
                breach_stats["budget_consumed_pct"].clip(0, 1000),
                color=budget_colors,
                alpha=0.85, width=20)

ax3.axhline(y=100, color="#D85A30", linewidth=2,
            linestyle="--",
            label="Budget exhausted (100%)")
ax3.axhline(y=50, color="#EF9F27", linewidth=1.5,
            linestyle=":",
            label="Half budget consumed (50%)")

ax3.set_title(
    "Error Budget Consumption per Batch Size\n"
    "(100% = SLA promise fully broken at this batch size)",
    fontsize=11, fontweight="bold"
)
ax3.set_ylabel("Budget Consumed (%)")
ax3.set_xticks(batch)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis="y")

# ── PLOT 4 — p95 latency with SLA line ──
ax4 = axes[3]

for _, row in breach_stats.iterrows():
    ax4.axvspan(
        row["batch_size"] - 8,
        row["batch_size"] + 8,
        alpha=0.15,
        color=risk_colors[row["breach_class"]]
    )

ax4.plot(batch, breach_stats["avg_p95_ms"],
         color="#EF9F27", linewidth=2.5,
         marker="s", markersize=6,
         label="Avg p95 latency")
ax4.plot(batch, breach_stats["max_p95_ms"],
         color="#D85A30", linewidth=2,
         marker="^", markersize=5,
         linestyle="--",
         label="Max p95 latency (worst run)")
ax4.plot(batch, breach_stats["avg_latency_ms"],
         color="#1D9E75", linewidth=2,
         marker="o", markersize=5,
         linestyle=":",
         label="Avg latency")

ax4.axhline(y=SLA_THRESHOLD_MS, color="red",
            linewidth=2, linestyle="-.",
            label="SLA threshold (20ms)")

ax4.set_title(
    "p95 Latency vs SLA Threshold — Breach Pattern",
    fontsize=11, fontweight="bold"
)
ax4.set_ylabel("Latency (ms)")
ax4.set_xlabel("Batch size")
ax4.set_xticks(batch)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# ── LEGEND ──
legend_elements = [
    Patch(facecolor="#1D9E75", alpha=0.5,
          label="Safe (breach rate < 10%)"),
    Patch(facecolor="#EF9F27", alpha=0.5,
          label="Warning (breach rate 10-50%)"),
    Patch(facecolor="#D85A30", alpha=0.5,
          label="Critical (breach rate > 50%)"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center", ncol=3,
    fontsize=10, bbox_to_anchor=(0.5, 0.01)
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    "/Users/devansh/Desktop/optimizeation/day12_breach_analysis.png",
    dpi=150, bbox_inches="tight"
)
print("✅ Saved → day12_breach_analysis.png")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("\n── Day 12 Findings ──\n")
print(breach_stats[[
    "batch_size", "total_runs", "breach_count",
    "breach_rate_pct", "breach_severity_ms",
    "budget_consumed_pct", "breach_class"
]].to_string(index=False))

print("\n── Breach Summary ──\n")
for level in ["Safe", "Warning", "Critical"]:
    batches = breach_stats[
        breach_stats["breach_class"] == level
    ]["batch_size"].tolist()
    print(f"  {level:<10} → batch sizes {batches}")

total_breaches = breach_stats["breach_count"].sum()
total_runs = breach_stats["total_runs"].sum()
overall_rate = (total_breaches / total_runs * 100).round(1)
print(f"\n  Overall breach rate : {overall_rate}%")
print(f"  Total breaches      : {int(total_breaches)} / {int(total_runs)} runs")
print(f"\n── Error Budget Status ──\n")
print(f"  SLA allows          : 0.1% breach rate")
print(f"  Your overall rate   : {overall_rate}%")
print(f"  Budget status       : {'EXCEEDED' if overall_rate > 0.1 else 'WITHIN BUDGET'}")