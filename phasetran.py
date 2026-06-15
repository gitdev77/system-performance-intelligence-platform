import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# DAY 22 — PHASE TRANSITION ANALYSIS
# Physics Layer, Day 2
#
# GOAL: Find the EXACT critical batch size (b_c)
# where the system flips from "ordered" (healthy)
# to "disordered" (collapsed) — using calculus,
# not just eyeballing a chart.
# ─────────────────────────────────────────

DATA_FILE  = "/Users/devansh/Desktop/optimizeation/telemetry_v2.csv"
OUTPUT_PNG = "/Users/devansh/Desktop/optimizeation/day22_phase_transition.png"

# ── STEP 1: Load & clean (same as Day 19-21) ──
df = pd.read_csv(DATA_FILE)
df["cpu_usage"] = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage"] = (
    df["cpu_usage"]
    .rolling(window=5, min_periods=1, center=True)
    .median().ffill().bfill()
)
df["avg_latency_ms"] = df["avg_latency"] * 1000

# ── STEP 2: Aggregate per batch size ──
agg = (
    df.groupby("batch_size")
    .agg(
        avg_latency_ms=("avg_latency_ms", "mean"),
        throughput=("throughput", "mean"),
        cv_latency=("cv_latency", "mean"),
    )
    .reset_index()
    .sort_values("batch_size")
    .reset_index(drop=True)
)

# ── STEP 3: Build the order parameter Φ(b) ──
throughput_peak = agg["throughput"].max()

agg["efficiency"] = agg["throughput"] / throughput_peak   # 1 = best ever
agg["stability"]  = 1 - agg["cv_latency"]                  # 1 = perfectly steady
agg["order_param"] = agg["efficiency"] * agg["stability"]  # Φ(b)

print("✅ Step 3 — Order parameter Φ(b) built (efficiency × stability)\n")

# ── STEP 4: Find the critical point b_c ──
# dΦ/db = rate of change of order parameter
# (np.gradient handles uneven batch-size spacing for us)
agg["d_order"] = np.gradient(agg["order_param"], agg["batch_size"])

# b_c = the batch size where Φ is FALLING fastest
# (most negative slope = steepest drop = phase boundary)
critical_idx = agg["d_order"].idxmin()
b_c = agg.loc[critical_idx, "batch_size"]

print(f"✅ Step 4 — Critical point found: b_c = {int(b_c)}")
print(f"   (steepest drop in Φ happens between the "
      f"neighbours of batch={int(b_c)})\n")

# ── STEP 5: Classify every batch into a PHASE ──
# "ordered"    = before the critical point (system healthy)
# "disordered" = at/after the critical point (system collapsed)
agg["phase"] = np.where(agg["batch_size"] < b_c, "ordered", "disordered")

n_ordered    = (agg["phase"] == "ordered").sum()
n_disordered = (agg["phase"] == "disordered").sum()

print(f"✅ Step 5 — Phase classification")
print(f"   Ordered phase    : {n_ordered} batch sizes "
      f"(batch < {int(b_c)})")
print(f"   Disordered phase : {n_disordered} batch sizes "
      f"(batch >= {int(b_c)})\n")

# ── STEP 6: Power-law fit (critical exponent β) ──
# Only makes sense INSIDE the ordered phase, where Φ
# is still decreasing TOWARD the critical point.
ordered = agg[agg["batch_size"] < b_c].copy()
ordered["dist_to_critical"] = b_c - ordered["batch_size"]

# avoid log(0) — keep only rows where both values are positive
valid = ordered[
    (ordered["dist_to_critical"] > 0) & (ordered["order_param"] > 0)
]

if len(valid) >= 3:
    log_x = np.log(valid["dist_to_critical"])
    log_y = np.log(valid["order_param"])

    # polyfit degree 1 = straight line: log(Φ) = β·log(dist) + log(A)
    beta, log_A = np.polyfit(log_x, log_y, 1)

    # r² = how well the straight line fits (1.0 = perfect)
    pred = beta * log_x + log_A
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
else:
    beta, log_A, r2 = np.nan, np.nan, np.nan

print(f"✅ Step 6 — Power-law fit (ordered phase only)")
if not np.isnan(beta):
    print(f"   Critical exponent β = {beta:.3f}")
    print(f"   Fit quality r²      = {r2:.3f}\n")
else:
    print(f"   Not enough points in ordered phase for a fit\n")

# ─────────────────────────────────────────
# STEP 7 — PLOT (4 panels)
# ─────────────────────────────────────────
colors = {
    "ordered"    : "#1D9E75",
    "disordered" : "#D85A30",
    "order_param": "#534AB7",
    "derivative" : "#EF9F27",
    "critical"   : "#E24B4A",
}

fig, axes = plt.subplots(4, 1, figsize=(13, 18))
fig.suptitle(
    "Day 22 — Phase Transition Analysis (Physics Layer)\n"
    "System Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

batch_arr = agg["batch_size"].values

# ── PANEL 1 — Order parameter Φ(b) ──
ax1 = axes[0]
ax1.plot(
    batch_arr, agg["order_param"],
    color=colors["order_param"], linewidth=2.5,
    marker="o", markersize=6, label="Order parameter Φ(b)"
)
ax1.axvline(
    x=b_c, color=colors["critical"], linewidth=2,
    linestyle="--", label=f"Critical point b_c = {int(b_c)}"
)
ax1.set_title(
    "Order Parameter Φ(b) = Efficiency × Stability\n"
    "(Φ near 1 = healthy, Φ dropping = system losing order)",
    fontsize=11, fontweight="bold"
)
ax1.set_xlabel("Batch size (control parameter)")
ax1.set_ylabel("Φ(b)")
ax1.set_xticks(batch_arr)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── PANEL 2 — dΦ/db (rate of change) ──
ax2 = axes[1]
bar_colors2 = [
    colors["critical"] if i == critical_idx else colors["derivative"]
    for i in range(len(agg))
]
ax2.bar(batch_arr, agg["d_order"], color=bar_colors2, alpha=0.85, width=20)
ax2.axhline(y=0, color="black", linewidth=1)
ax2.set_title(
    "Rate of Change dΦ/db — Where Does Order Fall Fastest?\n"
    "(most negative bar = the critical point, highlighted in red)",
    fontsize=11, fontweight="bold"
)
ax2.set_xlabel("Batch size")
ax2.set_ylabel("dΦ/db")
ax2.set_xticks(batch_arr)
ax2.grid(True, alpha=0.3, axis="y")

# ── PANEL 3 — Phase map ──
ax3 = axes[2]
bar_colors3 = [colors[p] for p in agg["phase"]]
ax3.bar(batch_arr, agg["order_param"], color=bar_colors3, alpha=0.85, width=20)
ax3.axvline(x=b_c, color=colors["critical"], linewidth=2, linestyle="--")

import matplotlib.patches as mpatches
legend_patches3 = [
    mpatches.Patch(color=colors["ordered"], label="Ordered phase (healthy)"),
    mpatches.Patch(color=colors["disordered"], label="Disordered phase (collapsed)"),
]
ax3.legend(handles=legend_patches3, fontsize=9)
ax3.set_title(
    "Phase Map — Ordered vs Disordered\n"
    f"(boundary at b_c = {int(b_c)})",
    fontsize=11, fontweight="bold"
)
ax3.set_xlabel("Batch size")
ax3.set_ylabel("Φ(b)")
ax3.set_xticks(batch_arr)
ax3.grid(True, alpha=0.3, axis="y")

# ── PANEL 4 — Power-law fit (log-log) ──
ax4 = axes[3]
if not np.isnan(beta):
    ax4.scatter(
        log_x, log_y, color=colors["order_param"], s=80,
        edgecolors="black", linewidth=0.5, zorder=5,
        label="Ordered-phase data points"
    )
    trend_x = np.linspace(log_x.min(), log_x.max(), 50)
    trend_y = beta * trend_x + log_A
    ax4.plot(
        trend_x, trend_y, color=colors["critical"],
        linewidth=2, linestyle="--",
        label=f"Power-law fit: β = {beta:.3f}, r² = {r2:.3f}"
    )
    for _, row in valid.iterrows():
        ax4.annotate(
            f"  {int(row['batch_size'])}",
            (np.log(row["dist_to_critical"]), np.log(row["order_param"])),
            fontsize=8
        )
    ax4.set_xlabel("log(b_c − b)  — log distance from critical point")
    ax4.set_ylabel("log(Φ(b))")
    ax4.legend(fontsize=9)
else:
    ax4.text(0.5, 0.5, "Not enough points for power-law fit",
             ha="center", va="center", fontsize=12)

ax4.set_title(
    "Critical Exponent Fit (log-log)\n"
    "(slope of this line = β, the critical exponent)",
    fontsize=11, fontweight="bold"
)
ax4.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"✅ Chart saved → {OUTPUT_PNG}\n")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("=" * 65)
print("  DAY 22 FINDINGS — PHASE TRANSITION ANALYSIS")
print("=" * 65)

print(f"\n── Order Parameter Table ──\n")
print(f"  {'Batch':<8} {'Efficiency':<12} {'Stability':<12} "
      f"{'Φ(b)':<10} {'dΦ/db':<10} {'Phase'}")
print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*12}")
for _, row in agg.iterrows():
    flag = " ⚠️ b_c" if row["batch_size"] == b_c else ""
    print(f"  {int(row['batch_size']):<8} "
          f"{row['efficiency']:<12.3f} "
          f"{row['stability']:<12.3f} "
          f"{row['order_param']:<10.3f} "
          f"{row['d_order']:<10.4f} "
          f"{row['phase']}{flag}")

print(f"\n── Critical Point ──\n")
print(f"  b_c = {int(b_c)}")
print(f"  Recall: Day 21's empirical collapse point was batch=192")
print(f"  Day 21's theory-predicted critical ρ never crossed 0.85")

print(f"\n── Critical Exponent ──\n")
if not np.isnan(beta):
    print(f"  β = {beta:.3f}  (r² = {r2:.3f})")
    if r2 > 0.8:
        print(f"  → STRONG power-law fit. Order parameter follows")
        print(f"    Φ ≈ A·(b_c - b)^{beta:.3f} in the ordered phase.")
    else:
        print(f"  → WEAK power-law fit (r²={r2:.3f}). Order parameter")
        print(f"    doesn't cleanly follow a power law with this")
        print(f"    little data — note this honestly in the writeup.")
else:
    print(f"  Not enough ordered-phase points to fit.")

print(f"\n{'='*65}")
print(f"  Day 22 complete. Output → day22_phase_transition.png")
print(f"{'='*65}\n")