import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import json
from datetime import datetime

# ─────────────────────────────────────────
# DAY 31 — OPTIMIZATION RECOMMENDATION ENGINE
# Cost Modeling Layer, Day 2
#
# GOAL: Synthesize Days 20-30 findings into
# one clean, actionable recommendation.
# Which batch size should production use?
# Why? What does it save? What's the risk?
# ─────────────────────────────────────────

DATA_FILE   = "/Users/devansh/Desktop/optimizeation/telemetry_v2.csv"
COST_FILE   = "/Users/devansh/Desktop/optimizeation/day30_cost_report.json"
OUTPUT_PNG  = "/Users/devansh/Desktop/optimizeation/day31_recommendation.png"
OUTPUT_JSON = "/Users/devansh/Desktop/optimizeation/day31_recommendation.json"

SLA_THRESHOLD_MS = 20.0
CPU_COST_PER_SEC = 0.05 / 3600
SLA_PENALTY      = 0.50

# ── THREE BUSINESS SCENARIOS ──
# Different businesses prioritize differently.
# weights = (latency_weight, cost_weight, sla_weight)
SCENARIOS = {
    "Latency-First\n(real-time API)" : (0.6, 0.2, 0.2),
    "Cost-First\n(batch pipeline)"   : (0.2, 0.6, 0.2),
    "SLA-First\n(regulated system)"  : (0.2, 0.2, 0.6),
}

# ─────────────────────────────────────────
# STEP 1 — Load & prepare
# ─────────────────────────────────────────
df = pd.read_csv(DATA_FILE)

df["cpu_usage_clean"] = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage_clean"] = (
    df["cpu_usage_clean"]
    .rolling(window=5, min_periods=1, center=True)
    .median().ffill().bfill()
)
df["avg_latency_ms"] = df["avg_latency"] * 1000
df["sla_breach_new"] = (
    df["avg_latency_ms"] > SLA_THRESHOLD_MS
).astype(int)

agg = (
    df.groupby("batch_size")
    .agg(
        avg_latency_ms  = ("avg_latency_ms", "mean"),
        throughput      = ("throughput",     "mean"),
        cpu_usage       = ("cpu_usage_clean","mean"),
        sla_breach_rate = ("sla_breach_new", "mean"),
        cv_latency      = ("cv_latency",     "mean"),
    )
    .reset_index()
    .sort_values("batch_size")
    .reset_index(drop=True)
)

agg["sla_breach_pct"] = agg["sla_breach_rate"] * 100

# ─────────────────────────────────────────
# STEP 2 — Recompute cost metrics
# (same as Day 30)
# ─────────────────────────────────────────
agg["latency_sec"]       = agg["avg_latency_ms"] / 1000
agg["cost_per_call"]     = agg["latency_sec"] * CPU_COST_PER_SEC
agg["cost_per_request"]  = agg["cost_per_call"] / agg["batch_size"]
agg["penalty_per_req"]   = (
    agg["sla_breach_rate"] * SLA_PENALTY / agg["batch_size"]
)
agg["total_cost"]        = (
    agg["cost_per_request"] + agg["penalty_per_req"]
)
throughput_peak          = agg["throughput"].max()
agg["efficiency"]        = agg["throughput"] / throughput_peak

print(f"✅ Steps 1-2 — Data loaded, costs computed\n")

# ─────────────────────────────────────────
# STEP 3 — PARETO FRONTIER ANALYSIS
#
# A batch size is "dominated" if another
# batch size beats it on BOTH latency AND
# cost simultaneously. Dominated points are
# never the right choice — something better
# exists in every dimension.
#
# We filter to only non-dominated points.
# ─────────────────────────────────────────
def is_dominated(idx, df_check):
    """
    Returns True if row at idx is dominated
    by at least one other row — meaning another
    row has BOTH lower latency AND lower cost.
    """
    row = df_check.iloc[idx]
    for j, other in df_check.iterrows():
        if j == df_check.index[idx]:
            continue
        if (other["avg_latency_ms"] <= row["avg_latency_ms"] and
                other["total_cost"] <= row["total_cost"] and
                (other["avg_latency_ms"] < row["avg_latency_ms"] or
                 other["total_cost"] < row["total_cost"])):
            return True
    return False

agg_reset = agg.reset_index(drop=True)
agg_reset["dominated"] = [
    is_dominated(i, agg_reset)
    for i in range(len(agg_reset))
]

pareto_df = agg_reset[~agg_reset["dominated"]].copy()
dominated_df = agg_reset[agg_reset["dominated"]].copy()

print(f"✅ Step 3 — Pareto frontier computed")
print(f"   Non-dominated (frontier) : "
      f"{len(pareto_df)} batch sizes "
      f"{pareto_df['batch_size'].tolist()}")
print(f"   Dominated (never optimal): "
      f"{len(dominated_df)} batch sizes "
      f"{dominated_df['batch_size'].tolist()}\n")

# ─────────────────────────────────────────
# STEP 4 — NORMALIZE FOR SCORING
#
# All three metrics (latency, cost, SLA) need
# to be on a 0-1 scale before weighting.
# 0 = worst, 1 = best (so we flip cost/latency
# since lower is better for those)
# ─────────────────────────────────────────
def normalize_col(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([1.0] * len(series),
                         index=series.index)
    return (series - mn) / (mx - mn)

# For latency and cost: lower = better
# so normalized_score = 1 - normalized_value
agg_reset["lat_score"]  = (
    1 - normalize_col(agg_reset["avg_latency_ms"])
)
agg_reset["cost_score"] = (
    1 - normalize_col(agg_reset["total_cost"])
)
agg_reset["sla_score"]  = (
    1 - normalize_col(agg_reset["sla_breach_rate"])
)

# ─────────────────────────────────────────
# STEP 5 — WEIGHTED SCORING (3 SCENARIOS)
#
# For each scenario, compute a weighted sum
# of the three normalized scores. The batch
# size with the highest total score wins
# under that scenario's priorities.
# ─────────────────────────────────────────
scenario_results = {}

print(f"✅ Step 5 — Weighted scoring across scenarios\n")

for scenario_name, (w_lat, w_cost, w_sla) in SCENARIOS.items():
    agg_reset[f"score_{scenario_name}"] = (
        w_lat  * agg_reset["lat_score"] +
        w_cost * agg_reset["cost_score"] +
        w_sla  * agg_reset["sla_score"]
    )
    best_idx   = agg_reset[
        f"score_{scenario_name}"
    ].idxmax()
    best_batch = int(agg_reset.loc[
        best_idx, "batch_size"
    ])
    best_score = agg_reset.loc[
        best_idx, f"score_{scenario_name}"
    ]

    # Safety margin: distance to nearest
    # batch size with sla_breach_pct > 0
    dangerous = agg_reset[
        agg_reset["sla_breach_pct"] > 0
    ]["batch_size"].values
    if len(dangerous) > 0:
        safety_margin = min(
            abs(best_batch - d) for d in dangerous
        )
    else:
        safety_margin = 999

    scenario_results[scenario_name] = {
        "recommended_batch": best_batch,
        "score"            : round(best_score, 4),
        "latency_ms"       : round(
            agg_reset.loc[
                best_idx, "avg_latency_ms"
            ], 2),
        "cost_per_req_usd" : round(
            agg_reset.loc[
                best_idx, "total_cost"
            ], 10),
        "sla_breach_pct"   : round(
            agg_reset.loc[
                best_idx, "sla_breach_pct"
            ], 1),
        "safety_margin"    : int(safety_margin),
        "weights"          : {
            "latency": w_lat,
            "cost"   : w_cost,
            "sla"    : w_sla
        }
    }

    clean_name = scenario_name.replace("\n", " ")
    print(f"  {clean_name}")
    print(f"    → Recommended batch : {best_batch}")
    print(f"    → Score             : {best_score:.4f}")
    print(f"    → Latency           : "
          f"{scenario_results[scenario_name]['latency_ms']}ms")
    print(f"    → Safety margin     : "
          f"{safety_margin} batch units\n")

# ─────────────────────────────────────────
# STEP 6 — FINAL RECOMMENDATION
#
# Take the most commonly recommended batch
# across all three scenarios. If all three
# agree — high confidence. If split — report
# both with context.
# ─────────────────────────────────────────
all_recommendations = [
    v["recommended_batch"]
    for v in scenario_results.values()
]
from collections import Counter
rec_counts   = Counter(all_recommendations)
final_batch  = rec_counts.most_common(1)[0][0]
consensus    = rec_counts[final_batch]
final_row    = agg_reset[
    agg_reset["batch_size"] == final_batch
].iloc[0]

print(f"✅ Step 6 — Final recommendation: "
      f"batch={final_batch} "
      f"({consensus}/3 scenarios agree)\n")

# ─────────────────────────────────────────
# STEP 7 — COMPUTE ANNUAL SAVINGS
# ─────────────────────────────────────────
worst_cost  = agg_reset["total_cost"].max()
best_cost   = final_row["total_cost"]
daily_reqs  = 1_000_000

daily_saving  = (worst_cost - best_cost) * daily_reqs
annual_saving = daily_saving * 365

# ─────────────────────────────────────────
# STEP 8 — PLOT (4 panels)
# ─────────────────────────────────────────
colors = {
    "frontier"  : "#1D9E75",
    "dominated" : "#999999",
    "optimal"   : "#E24B4A",
    "scenario_A": "#534AB7",
    "scenario_B": "#EF9F27",
    "scenario_C": "#D85A30",
    "safe"      : "#1D9E75",
    "danger"    : "#D85A30",
}

fig, axes = plt.subplots(4, 1, figsize=(13, 18))
fig.suptitle(
    "Day 31 — Optimization Recommendation Engine\n"
    "System Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

batch_arr = agg_reset["batch_size"].values

# ── PANEL 1 — Pareto frontier plot ──
ax1 = axes[0]

ax1.scatter(
    dominated_df["avg_latency_ms"],
    dominated_df["total_cost"] * 1e6,
    color=colors["dominated"], s=80,
    alpha=0.6, label="Dominated (never optimal)",
    zorder=3
)
ax1.scatter(
    pareto_df["avg_latency_ms"],
    pareto_df["total_cost"] * 1e6,
    color=colors["frontier"], s=120,
    alpha=0.9, label="Pareto frontier",
    zorder=4, edgecolors="black", linewidth=0.5
)
ax1.scatter(
    [final_row["avg_latency_ms"]],
    [final_row["total_cost"] * 1e6],
    color=colors["optimal"], s=200,
    zorder=5, marker="*",
    label=f"Final recommendation "
          f"(batch={final_batch})"
)

for _, row in agg_reset.iterrows():
    ax1.annotate(
        f"  {int(row['batch_size'])}",
        (row["avg_latency_ms"],
         row["total_cost"] * 1e6),
        fontsize=7
    )

ax1.axvline(
    x=SLA_THRESHOLD_MS,
    color=colors["danger"], linewidth=2,
    linestyle="--",
    label=f"SLA limit ({SLA_THRESHOLD_MS}ms)"
)

ax1.set_title(
    "Pareto Frontier — Cost vs Latency\n"
    "(green = genuinely good options, "
    "gray = dominated/never optimal)",
    fontsize=11, fontweight="bold"
)
ax1.set_xlabel("Avg Latency (ms)")
ax1.set_ylabel("Total cost per request (μ$)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# ── PANEL 2 — Scenario scores ──
ax2 = axes[1]

scenario_colors = [
    colors["scenario_A"],
    colors["scenario_B"],
    colors["scenario_C"]
]
scenario_names_clean = [
    s.replace("\n", " ") for s in SCENARIOS.keys()
]
score_cols = [
    f"score_{s}" for s in SCENARIOS.keys()
]

x     = np.arange(len(batch_arr))
width = 0.28

for i, (col, sc_name, sc_color) in enumerate(zip(
    score_cols, scenario_names_clean, scenario_colors
)):
    ax2.bar(
        x + (i - 1) * width,
        agg_reset[col],
        width, color=sc_color,
        alpha=0.8, label=sc_name
    )

ax2.set_xticks(x)
ax2.set_xticklabels(batch_arr, fontsize=7)
ax2.set_title(
    "Weighted Score Per Batch Size — Three Business Scenarios\n"
    "(higher bar = better choice under that scenario's priorities)",
    fontsize=11, fontweight="bold"
)
ax2.set_ylabel("Weighted score (0-1)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis="y")

# ── PANEL 3 — Safety zone map ──
ax3 = axes[2]

bar_colors3 = [
    colors["danger"] if r["sla_breach_pct"] > 0
    else colors["safe"]
    for _, r in agg_reset.iterrows()
]
ax3.bar(
    batch_arr,
    agg_reset["sla_breach_pct"],
    color=bar_colors3, alpha=0.85, width=20
)

ax3.axvline(
    x=final_batch,
    color=colors["optimal"], linewidth=3,
    linestyle="--",
    label=f"Recommended: batch={final_batch}"
)

safe_patch   = mpatches.Patch(
    color=colors["safe"], label="Safe (0% breach)"
)
danger_patch = mpatches.Patch(
    color=colors["danger"], label="Dangerous (>0% breach)"
)
ax3.legend(
    handles=[safe_patch, danger_patch,
             mpatches.Patch(color=colors["optimal"],
                            label=f"Recommended batch={final_batch}")],
    fontsize=8
)
ax3.set_title(
    "Safety Zone Map — SLA Breach Rate Per Batch Size\n"
    "(red = avoid | green = safe | "
    "dashed line = recommendation)",
    fontsize=11, fontweight="bold"
)
ax3.set_ylabel("SLA breach rate (%)")
ax3.set_xlabel("Batch size")
ax3.set_xticks(batch_arr)
ax3.grid(True, alpha=0.3, axis="y")

# ── PANEL 4 — Final recommendation summary ──
ax4 = axes[3]
ax4.axis("off")

summary_text = f"""
FINAL RECOMMENDATION
{'─'*55}

Recommended Batch Size  :  {final_batch}
Scenarios In Agreement  :  {consensus} / 3

Performance at batch={final_batch}:
  Avg Latency           :  {final_row['avg_latency_ms']:.2f} ms
  SLA Breach Rate       :  {final_row['sla_breach_pct']:.1f}%
  Throughput Efficiency :  {final_row['efficiency']*100:.1f}% of peak
  Total Cost/Request    :  {final_row['total_cost']*1e6:.4f} μ$

Business Impact (vs worst-case batch):
  Daily saving (1M req) :  ${daily_saving:,.2f}
  Annual saving         :  ${annual_saving:,.2f}

Safety:
  Nearest danger zone   :  batch={min(dangerous) if len(dangerous)>0 else 'N/A'}
  Safety margin         :  {scenario_results[list(SCENARIOS.keys())[0]]['safety_margin']} batch units

Verdict: RUN AT BATCH={final_batch}
"""

ax4.text(
    0.05, 0.95, summary_text,
    transform=ax4.transAxes,
    fontsize=11, verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(
        boxstyle="round", facecolor="#1D9E75",
        alpha=0.1, edgecolor="#1D9E75",
        linewidth=2
    )
)

ax4.set_title(
    "Final Recommendation Summary",
    fontsize=11, fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"✅ Chart saved → {OUTPUT_PNG}\n")

# ─────────────────────────────────────────
# STEP 9 — SAVE RECOMMENDATION JSON
# ─────────────────────────────────────────
recommendation_output = {
    "generated_at"       : datetime.now().isoformat(),
    "final_recommendation": {
        "batch_size"         : final_batch,
        "consensus_scenarios": f"{consensus}/3",
        "latency_ms"         : round(
            final_row["avg_latency_ms"], 2),
        "sla_breach_pct"     : round(
            final_row["sla_breach_pct"], 1),
        "total_cost_usd"     : round(
            final_row["total_cost"], 10),
        "efficiency_pct"     : round(
            final_row["efficiency"] * 100, 1),
        "annual_saving_usd"  : round(annual_saving, 2),
        "daily_saving_usd"   : round(daily_saving, 2),
    },
    "pareto_frontier_batches": pareto_df[
        "batch_size"
    ].tolist(),
    "dominated_batches"      : dominated_df[
        "batch_size"
    ].tolist(),
    "scenario_results"       : {
        k.replace("\n", " "): v
        for k, v in scenario_results.items()
    },
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(recommendation_output, f, indent=2)

print(f"✅ Recommendation saved → {OUTPUT_JSON}\n")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("=" * 65)
print("  DAY 31 FINDINGS — OPTIMIZATION RECOMMENDATION")
print("=" * 65)

print(f"\n── Pareto Frontier ──\n")
print(f"  Non-dominated batch sizes: "
      f"{pareto_df['batch_size'].tolist()}")
print(f"  Dominated (avoid)        : "
      f"{dominated_df['batch_size'].tolist()}")

print(f"\n── Scenario Recommendations ──\n")
for name, result in scenario_results.items():
    clean = name.replace('\n', ' ')
    print(f"  {clean}")
    print(f"    Recommended : batch="
          f"{result['recommended_batch']}")
    print(f"    Score       : {result['score']}")
    print(f"    Latency     : {result['latency_ms']}ms")
    print(f"    SLA breach  : "
          f"{result['sla_breach_pct']}%\n")

print(f"── Final Recommendation ──\n")
print(f"  BATCH = {final_batch} "
      f"({consensus}/3 scenarios agree)")
print(f"  Latency   : {final_row['avg_latency_ms']:.2f}ms")
print(f"  Breach    : {final_row['sla_breach_pct']:.1f}%")
print(f"  Efficiency: {final_row['efficiency']*100:.1f}% of peak")

print(f"\n── Business Impact ──\n")
print(f"  Annual saving vs worst case: "
      f"${annual_saving:,.2f}")
print(f"  Daily saving (1M requests) : "
      f"${daily_saving:,.2f}")

print(f"\n{'='*65}")
print(f"  Day 31 complete.")
print(f"  → day31_recommendation.png")
print(f"  → day31_recommendation.json")
print(f"{'='*65}\n")