import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import json
from datetime import datetime

# ─────────────────────────────────────────
# DAY 30 — COST MODELING LAYER, DAY 1
# Cost Per Request Analysis
#
# GOAL: Translate raw telemetry into dollar
# costs — finding which batch size is actually
# cheapest per individual inference request,
# once you factor in SLA penalty costs too.
# ─────────────────────────────────────────

DATA_FILE  = "/Users/devansh/Desktop/optimizeation/telemetry_v2.csv"
OUTPUT_PNG = "/Users/devansh/Desktop/optimizeation/day30_cost_modeling.png"
OUTPUT_JSON= "/Users/devansh/Desktop/optimizeation/day30_cost_report.json"

SLA_THRESHOLD_MS = 20.0

# ── COST PARAMETERS ──
# These are configurable — swap for real
# cloud provider rates if needed
CPU_COST_PER_HOUR   = 0.05    # $ per CPU-hour (mid-tier on-demand)
CPU_COST_PER_SEC    = CPU_COST_PER_HOUR / 3600
SLA_PENALTY_PER_BREACH = 0.50 # $ penalty per SLA breach event

print(f"✅ Cost parameters:")
print(f"   CPU cost  : ${CPU_COST_PER_HOUR}/hour = "
      f"${CPU_COST_PER_SEC:.8f}/second")
print(f"   SLA penalty: ${SLA_PENALTY_PER_BREACH} per breach\n")

# ─────────────────────────────────────────
# STEP 1 — Load & clean
# ─────────────────────────────────────────
df = pd.read_csv(DATA_FILE)

df["cpu_usage_clean"] = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage_clean"] = (
    df["cpu_usage_clean"]
    .rolling(window=5, min_periods=1, center=True)
    .median().ffill().bfill()
)
df["avg_latency_ms"] = df["avg_latency"] * 1000
df["p95_latency_ms"] = df["p95_latency"] * 1000
df["sla_breach_new"] = (
    df["avg_latency_ms"] > SLA_THRESHOLD_MS
).astype(int)

# ─────────────────────────────────────────
# STEP 2 — Aggregate per batch size
# ─────────────────────────────────────────
agg = (
    df.groupby("batch_size")
    .agg(
        avg_latency_ms  = ("avg_latency_ms",  "mean"),
        p95_latency_ms  = ("p95_latency_ms",  "mean"),
        throughput      = ("throughput",       "mean"),
        cpu_usage       = ("cpu_usage_clean",  "mean"),
        sla_breach_rate = ("sla_breach_new",   "mean"),
        cv_latency      = ("cv_latency",       "mean"),
    )
    .reset_index()
    .sort_values("batch_size")
    .reset_index(drop=True)
)

agg["sla_breach_pct"] = agg["sla_breach_rate"] * 100

print(f"✅ Steps 1-2 — Data loaded and aggregated\n")

# ─────────────────────────────────────────
# STEP 3 — COMPUTE COST METRICS
# ─────────────────────────────────────────

# 3a. Latency in seconds (for cost formula)
agg["latency_sec"] = agg["avg_latency_ms"] / 1000

# 3b. Cost per inference CALL
# (one call processes batch_size requests)
agg["cost_per_call"] = (
    agg["latency_sec"] * CPU_COST_PER_SEC
)

# 3c. Cost per individual REQUEST
# (spread call cost across all requests in batch)
agg["cost_per_request"] = (
    agg["cost_per_call"] / agg["batch_size"]
)

# 3d. Expected SLA penalty per request
# breach_rate × penalty / batch_size
# (penalty is per-call so divide by batch to
#  get per-request impact)
agg["penalty_per_request"] = (
    agg["sla_breach_rate"] *
    SLA_PENALTY_PER_BREACH /
    agg["batch_size"]
)

# 3e. TOTAL cost per request
# (compute + penalty — the real business number)
agg["total_cost_per_request"] = (
    agg["cost_per_request"] +
    agg["penalty_per_request"]
)

# 3f. Throughput efficiency (from Day 22)
throughput_peak = agg["throughput"].max()
agg["efficiency"] = agg["throughput"] / throughput_peak

# 3g. Cost-efficiency score
# "how much value (throughput) per dollar spent?"
# Higher = better. Peak efficiency at lowest cost.
# We normalize both to 0-1 and take the ratio.
cost_max = agg["total_cost_per_request"].max()
cost_min = agg["total_cost_per_request"].min()
agg["cost_norm"] = (
    (agg["total_cost_per_request"] - cost_min) /
    (cost_max - cost_min)
)
# Cost-efficiency = efficiency / normalized cost
# (high throughput efficiency + low cost = best)
agg["cost_efficiency_score"] = (
    agg["efficiency"] /
    (agg["cost_norm"] + 0.01)  # +0.01 avoids divide by zero
)

# Find optimal batch size = highest cost-efficiency
optimal_idx   = agg["cost_efficiency_score"].idxmax()
optimal_batch = agg.loc[optimal_idx, "batch_size"]

# Cheapest raw compute (ignoring SLA penalty)
cheapest_compute_idx = agg["cost_per_request"].idxmin()
cheapest_batch       = agg.loc[
    cheapest_compute_idx, "batch_size"
]

print(f"✅ Step 3 — Cost metrics computed")
print(f"   Cheapest compute cost at  : "
      f"batch={int(cheapest_batch)}")
print(f"   Best cost-efficiency at   : "
      f"batch={int(optimal_batch)}\n")

# ─────────────────────────────────────────
# STEP 4 — PLOT (4 panels)
# ─────────────────────────────────────────
colors = {
    "compute" : "#534AB7",
    "penalty" : "#D85A30",
    "total"   : "#1D9E75",
    "optimal" : "#E24B4A",
    "score"   : "#EF9F27",
    "gray"    : "#999999",
}

fig, axes = plt.subplots(4, 1, figsize=(13, 18))
fig.suptitle(
    "Day 30 — Cost Modeling Layer\n"
    "Cost Per Request Analysis",
    fontsize=14, fontweight="bold", y=0.98
)

batch_arr = agg["batch_size"].values

# ── PANEL 1 — Cost breakdown (stacked bar) ──
# Shows compute cost + SLA penalty cost stacked
# so you see total AND what's driving it
ax1 = axes[0]

bars_compute = ax1.bar(
    batch_arr,
    agg["cost_per_request"] * 1e6,  # convert to micro-dollars
    color=colors["compute"], alpha=0.85,
    width=20, label="Compute cost"
)
bars_penalty = ax1.bar(
    batch_arr,
    agg["penalty_per_request"] * 1e6,
    bottom=agg["cost_per_request"] * 1e6,
    color=colors["penalty"], alpha=0.85,
    width=20, label="SLA penalty cost"
)

ax1.axvline(
    x=optimal_batch, color=colors["optimal"],
    linewidth=2, linestyle="--",
    label=f"Optimal batch (best efficiency) = {int(optimal_batch)}"
)

ax1.set_title(
    "Cost Per Request — Compute + SLA Penalty\n"
    "(stacked bars show what's driving total cost at each batch size)",
    fontsize=11, fontweight="bold"
)
ax1.set_ylabel("Cost per request (micro-dollars, μ$)")
ax1.set_xlabel("Batch size")
ax1.set_xticks(batch_arr)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis="y")

# ── PANEL 2 — Total cost curve ──
# The U-shaped (or downward) curve showing
# where cost per request is minimized
ax2 = axes[1]

ax2.plot(
    batch_arr,
    agg["total_cost_per_request"] * 1e6,
    color=colors["total"], linewidth=2.5,
    marker="o", markersize=6,
    label="Total cost per request"
)
ax2.plot(
    batch_arr,
    agg["cost_per_request"] * 1e6,
    color=colors["compute"], linewidth=1.5,
    linestyle="--", marker="s", markersize=4,
    alpha=0.7, label="Compute only (no penalty)"
)

# Highlight the minimum
min_cost = agg["total_cost_per_request"].min() * 1e6
ax2.axhline(
    y=min_cost,
    color=colors["optimal"], linewidth=1.5,
    linestyle=":", alpha=0.7,
    label=f"Minimum cost = {min_cost:.2f}μ$"
)
ax2.scatter(
    [optimal_batch],
    [agg.loc[optimal_idx, "total_cost_per_request"] * 1e6],
    color=colors["optimal"], s=150,
    zorder=5, marker="*",
    label=f"Optimal batch = {int(optimal_batch)}"
)

ax2.set_title(
    "Total Cost Per Request vs Batch Size\n"
    "(where is cost minimized?)",
    fontsize=11, fontweight="bold"
)
ax2.set_ylabel("Cost per request (μ$)")
ax2.set_xlabel("Batch size")
ax2.set_xticks(batch_arr)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ── PANEL 3 — Cost vs Latency scatter ──
# The classic tradeoff chart:
# x = latency (performance), y = cost
# Low-left = ideal (fast AND cheap)
# High-right = terrible (slow AND expensive)
ax3 = axes[2]

scatter = ax3.scatter(
    agg["avg_latency_ms"],
    agg["total_cost_per_request"] * 1e6,
    c=batch_arr, cmap="viridis",
    s=100, zorder=5,
    edgecolors="black", linewidth=0.5
)
plt.colorbar(scatter, ax=ax3, label="Batch size")

for _, row in agg.iterrows():
    ax3.annotate(
        f"  {int(row['batch_size'])}",
        (row["avg_latency_ms"],
         row["total_cost_per_request"] * 1e6),
        fontsize=8
    )

ax3.axvline(
    x=SLA_THRESHOLD_MS,
    color=colors["penalty"], linewidth=2,
    linestyle="--",
    label=f"SLA limit ({SLA_THRESHOLD_MS}ms)"
)

ax3.set_title(
    "Cost vs Latency Tradeoff\n"
    "(bottom-left = fast AND cheap = ideal operating zone)",
    fontsize=11, fontweight="bold"
)
ax3.set_xlabel("Avg Latency (ms)")
ax3.set_ylabel("Total cost per request (μ$)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ── PANEL 4 — Cost-efficiency score ──
ax4 = axes[3]

bar_colors4 = [
    colors["optimal"] if b == optimal_batch
    else colors["score"]
    for b in batch_arr
]
bars4 = ax4.bar(
    batch_arr, agg["cost_efficiency_score"],
    color=bar_colors4, alpha=0.85, width=20
)

ax4.set_title(
    "Cost-Efficiency Score Per Batch Size\n"
    "(higher = more throughput per dollar — "
    f"winner = batch {int(optimal_batch)} in red)",
    fontsize=11, fontweight="bold"
)
ax4.set_ylabel("Cost-efficiency score\n"
               "(throughput efficiency / normalized cost)")
ax4.set_xlabel("Batch size")
ax4.set_xticks(batch_arr)
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"✅ Chart saved → {OUTPUT_PNG}\n")

# ─────────────────────────────────────────
# STEP 5 — SAVE COST REPORT JSON
# ─────────────────────────────────────────
cost_report = {
    "generated_at"    : datetime.now().isoformat(),
    "cost_parameters" : {
        "cpu_cost_per_hour"      : CPU_COST_PER_HOUR,
        "cpu_cost_per_second"    : CPU_COST_PER_SEC,
        "sla_penalty_per_breach" : SLA_PENALTY_PER_BREACH,
        "currency"               : "USD",
    },
    "optimal_batch_size"  : int(optimal_batch),
    "cheapest_compute_batch": int(cheapest_batch),
    "per_batch_costs": [
        {
            "batch_size"           : int(r["batch_size"]),
            "avg_latency_ms"       : round(r["avg_latency_ms"], 2),
            "cost_per_call_usd"    : round(r["cost_per_call"], 8),
            "cost_per_request_usd" : round(r["cost_per_request"], 8),
            "penalty_per_request"  : round(r["penalty_per_request"], 8),
            "total_cost_per_request": round(
                r["total_cost_per_request"], 8),
            "sla_breach_pct"       : round(r["sla_breach_pct"], 1),
            "cost_efficiency_score": round(
                r["cost_efficiency_score"], 4),
        }
        for _, r in agg.iterrows()
    ]
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(cost_report, f, indent=2)

print(f"✅ Cost report saved → {OUTPUT_JSON}\n")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("=" * 70)
print("  DAY 30 FINDINGS — COST PER REQUEST ANALYSIS")
print("=" * 70)

print(f"\n── Full Cost Table ──\n")
print(f"  {'Batch':<8} {'Latency':<10} {'Compute':<12} "
      f"{'Penalty':<12} {'Total':<12} {'Breach%':<10} {'Score'}")
print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*12} "
      f"{'-'*12} {'-'*10} {'-'*8}")

for _, row in agg.iterrows():
    flag = " ⭐" if row["batch_size"] == optimal_batch else ""
    flag += " ⚠️" if row["sla_breach_pct"] > 0 else ""
    print(
        f"  {int(row['batch_size']):<8} "
        f"{row['avg_latency_ms']:.2f}ms    "
        f"{row['cost_per_request']*1e6:.3f}μ$      "
        f"{row['penalty_per_request']*1e6:.3f}μ$      "
        f"{row['total_cost_per_request']*1e6:.3f}μ$      "
        f"{row['sla_breach_pct']:.1f}%       "
        f"{row['cost_efficiency_score']:.3f}{flag}"
    )

print(f"\n── Key Findings ──\n")
best_row    = agg.loc[optimal_idx]
cheapest_row= agg.loc[cheapest_compute_idx]

print(f"  Optimal batch (cost-efficiency): "
      f"{int(optimal_batch)}")
print(f"    → Total cost  : "
      f"{best_row['total_cost_per_request']*1e6:.3f}μ$ per request")
print(f"    → Latency     : "
      f"{best_row['avg_latency_ms']:.2f}ms")
print(f"    → SLA breach  : "
      f"{best_row['sla_breach_pct']:.1f}%")

print(f"\n  Cheapest raw compute: batch={int(cheapest_batch)}")
print(f"    → Compute cost: "
      f"{cheapest_row['cost_per_request']*1e6:.3f}μ$ per request")
print(f"    → SLA breach  : "
      f"{cheapest_row['sla_breach_pct']:.1f}%")

# Cost comparison: optimal vs worst
worst_idx  = agg["total_cost_per_request"].idxmax()
worst_batch= agg.loc[worst_idx, "batch_size"]
worst_cost = agg.loc[worst_idx,
                     "total_cost_per_request"]
best_cost  = best_row["total_cost_per_request"]
savings_pct= ((worst_cost - best_cost) / worst_cost) * 100

print(f"\n  Most expensive batch: {int(worst_batch)}")
print(f"    → Total cost: "
      f"{worst_cost*1e6:.3f}μ$ per request")
print(f"\n  Choosing batch={int(optimal_batch)} over "
      f"batch={int(worst_batch)} saves "
      f"{savings_pct:.1f}% per request.")

print(f"\n── Business Interpretation ──\n")
print(f"  At 1 million requests/day:")
daily_best  = best_cost  * 1_000_000
daily_worst = worst_cost * 1_000_000
print(f"  Optimal batch  → ${daily_best:.2f}/day")
print(f"  Worst batch    → ${daily_worst:.2f}/day")
print(f"  Daily saving   → "
      f"${daily_worst - daily_best:.2f}/day")
print(f"  Annual saving  → "
      f"${(daily_worst-daily_best)*365:.2f}/year")

print(f"\n{'='*70}")
print(f"  Day 30 complete.")
print(f"  → day30_cost_modeling.png")
print(f"  → day30_cost_report.json")
print(f"{'='*70}\n")