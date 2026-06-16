import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# DAY 22B — REFINED ORDER PARAMETER
# Physics Layer, Day 2 (continued)
#
# GOAL: Day 22's Φ = efficiency × stability gave
# a noisy result (b_c = 32, not matching Days 20-21).
# Today: strip out the noisy "stability" term and
# use PURE efficiency as the order parameter.
# Compare both versions side by side.
# ─────────────────────────────────────────

DATA_FILE  = "/Users/devansh/Desktop/optimizeation/telemetry_v2.csv"
OUTPUT_PNG = "/Users/devansh/Desktop/optimizeation/day22b_order_param_refined.png"

# ── STEP 1: Load & clean (same as before) ──
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

throughput_peak = agg["throughput"].max()

# ── STEP 3: Build BOTH order parameters ──

# Version 1 (Day 22): combined — efficiency × stability
agg["efficiency"] = agg["throughput"] / throughput_peak
agg["stability"]  = 1 - agg["cv_latency"]
agg["phi_combined"] = agg["efficiency"] * agg["stability"]

# Version 2 (Day 22B): pure efficiency only
agg["phi_pure"] = agg["efficiency"]

print("✅ Step 3 — Built two order parameters: combined vs pure\n")

# ── STEP 4: Find critical point for EACH version ──
agg["d_combined"] = np.gradient(agg["phi_combined"], agg["batch_size"])
agg["d_pure"]     = np.gradient(agg["phi_pure"], agg["batch_size"])

idx_combined = agg["d_combined"].idxmin()
idx_pure     = agg["d_pure"].idxmin()

b_c_combined = agg.loc[idx_combined, "batch_size"]
b_c_pure     = agg.loc[idx_pure, "batch_size"]

print(f"✅ Step 4 — Critical points found")
print(f"   Combined Φ (Day 22)  → b_c = {int(b_c_combined)}")
print(f"   Pure Φ    (Day 22B)  → b_c = {int(b_c_pure)}")
print(f"   Day 21 empirical collapse point = batch 192\n")

# ── STEP 5: Phase classification using PURE Φ ──
agg["phase"] = np.where(agg["batch_size"] < b_c_pure, "ordered", "disordered")

n_ordered    = (agg["phase"] == "ordered").sum()
n_disordered = (agg["phase"] == "disordered").sum()

print(f"✅ Step 5 — Phase classification (using pure Φ)")
print(f"   Ordered phase    : {n_ordered} batch sizes "
      f"(batch < {int(b_c_pure)})")
print(f"   Disordered phase : {n_disordered} batch sizes "
      f"(batch >= {int(b_c_pure)})\n")

# ── STEP 6: Power-law fit with pure Φ ──
ordered = agg[agg["batch_size"] < b_c_pure].copy()
ordered["dist_to_critical"] = b_c_pure - ordered["batch_size"]

valid = ordered[
    (ordered["dist_to_critical"] > 0) & (ordered["phi_pure"] > 0)
]

if len(valid) >= 3:
    log_x = np.log(valid["dist_to_critical"])
    log_y = np.log(valid["phi_pure"])

    beta, log_A = np.polyfit(log_x, log_y, 1)

    pred = beta * log_x + log_A
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
else:
    beta, log_A, r2 = np.nan, np.nan, np.nan

print(f"✅ Step 6 — Power-law fit (pure Φ, ordered phase)")
if not np.isnan(beta):
    print(f"   Critical exponent β = {beta:.3f}")
    print(f"   Fit quality r²      = {r2:.3f}\n")
else:
    print(f"   Still not enough ordered-phase points ({len(valid)} found)\n")

# ─────────────────────────────────────────
# STEP 7 — PLOT (4 panels)
# ─────────────────────────────────────────
colors = {
    "combined"  : "#999999",
    "pure"      : "#1D9E75",
    "critical"  : "#E24B4A",
    "empirical" : "#534AB7",
    "ordered"   : "#1D9E75",
    "disordered": "#D85A30",
}

fig, axes = plt.subplots(4, 1, figsize=(13, 18))
fig.suptitle(
    "Day 22B — Refined Order Parameter (Pure Efficiency)\n"
    "System Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

batch_arr = agg["batch_size"].values

# ── PANEL 1 — Both order parameters overlaid ──
ax1 = axes[0]
ax1.plot(
    batch_arr, agg["phi_combined"],
    color=colors["combined"], linewidth=1.5, linestyle="--",
    marker="s", markersize=5, alpha=0.6,
    label="Φ_combined = efficiency × stability (Day 22)"
)
ax1.plot(
    batch_arr, agg["phi_pure"],
    color=colors["pure"], linewidth=2.5,
    marker="o", markersize=6,
    label="Φ_pure = efficiency only (Day 22B)"
)
ax1.axvline(
    x=b_c_pure, color=colors["critical"], linewidth=2,
    linestyle="--", label=f"Pure Φ critical point: b_c = {int(b_c_pure)}"
)
ax1.axvline(
    x=192, color=colors["empirical"], linewidth=2,
    linestyle=":", label="Day 21 empirical collapse: batch=192"
)
ax1.set_title(
    "Order Parameter — Combined (noisy) vs Pure Efficiency (clean)\n"
    "(does the cleaner version land closer to batch=192?)",
    fontsize=11, fontweight="bold"
)
ax1.set_xlabel("Batch size (control parameter)")
ax1.set_ylabel("Φ(b)")
ax1.set_xticks(batch_arr)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# ── PANEL 2 — dΦ/db for pure efficiency ──
ax2 = axes[1]
bar_colors2 = [
    colors["critical"] if i == idx_pure else "#EF9F27"
    for i in range(len(agg))
]
ax2.bar(batch_arr, agg["d_pure"], color=bar_colors2, alpha=0.85, width=20)
ax2.axhline(y=0, color="black", linewidth=1)
ax2.set_title(
    "Rate of Change dΦ_pure/db — Steepest Drop Highlighted\n"
    f"(critical point b_c = {int(b_c_pure)})",
    fontsize=11, fontweight="bold"
)
ax2.set_xlabel("Batch size")
ax2.set_ylabel("dΦ_pure/db")
ax2.set_xticks(batch_arr)
ax2.grid(True, alpha=0.3, axis="y")

# ── PANEL 3 — Phase map using pure Φ ──
ax3 = axes[2]
bar_colors3 = [colors[p] for p in agg["phase"]]
ax3.bar(batch_arr, agg["phi_pure"], color=bar_colors3, alpha=0.85, width=20)
ax3.axvline(x=b_c_pure, color=colors["critical"], linewidth=2, linestyle="--")

legend_patches3 = [
    mpatches.Patch(color=colors["ordered"], label="Ordered phase (healthy)"),
    mpatches.Patch(color=colors["disordered"], label="Disordered phase (collapsed)"),
]
ax3.legend(handles=legend_patches3, fontsize=9)
ax3.set_title(
    "Phase Map (Pure Efficiency) — Ordered vs Disordered\n"
    f"(boundary at b_c = {int(b_c_pure)})",
    fontsize=11, fontweight="bold"
)
ax3.set_xlabel("Batch size")
ax3.set_ylabel("Φ_pure(b)")
ax3.set_xticks(batch_arr)
ax3.grid(True, alpha=0.3, axis="y")

# ── PANEL 4 — Power-law fit (log-log) ──
ax4 = axes[3]
if not np.isnan(beta):
    ax4.scatter(
        log_x, log_y, color=colors["pure"], s=80,
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
            (np.log(row["dist_to_critical"]), np.log(row["phi_pure"])),
            fontsize=8
        )
    ax4.set_xlabel("log(b_c − b)  — log distance from critical point")
    ax4.set_ylabel("log(Φ_pure(b))")
    ax4.legend(fontsize=9)
else:
    ax4.text(0.5, 0.5, f"Not enough points for power-law fit "
                       f"({len(valid)} found, need ≥3)",
             ha="center", va="center", fontsize=12,
             transform=ax4.transAxes)

ax4.set_title(
    "Critical Exponent Fit (log-log, Pure Φ)\n"
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
print("  DAY 22B FINDINGS — REFINED ORDER PARAMETER")
print("=" * 65)

print(f"\n── Comparison Table ──\n")
print(f"  {'Batch':<8} {'Φ_combined':<12} {'Φ_pure':<10} "
      f"{'dΦ_pure/db':<12} {'Phase'}")
print(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*12} {'-'*12}")
for _, row in agg.iterrows():
    flag = " ⚠️ b_c" if row["batch_size"] == b_c_pure else ""
    print(f"  {int(row['batch_size']):<8} "
          f"{row['phi_combined']:<12.3f} "
          f"{row['phi_pure']:<10.3f} "
          f"{row['d_pure']:<12.4f} "
          f"{row['phase']}{flag}")

print(f"\n── Critical Point Comparison ──\n")
print(f"  Φ_combined (Day 22)  → b_c = {int(b_c_combined)}")
print(f"  Φ_pure     (Day 22B) → b_c = {int(b_c_pure)}")
print(f"  Empirical  (Day 20/21) → batch = 192")

diff = abs(b_c_pure - 192)
diff_pct = (diff / 192) * 100
print(f"\n  Pure Φ vs empirical difference: {diff:.0f} ({diff_pct:.0f}%)")
if diff_pct < 25:
    print(f"  ✅ CLOSE MATCH — pure efficiency order parameter agrees")
    print(f"     with the empirical collapse point within {diff_pct:.0f}%.")
else:
    print(f"  ⚠️  Still a gap. This is worth documenting honestly —")
    print(f"     'steepest drop' and 'sustained collapse' can be")
    print(f"     genuinely different points in a noisy real system.")

print(f"\n── Critical Exponent (Pure Φ) ──\n")
if not np.isnan(beta):
    print(f"  β = {beta:.3f}  (r² = {r2:.3f})")
else:
    print(f"  Not enough ordered-phase points to fit "
          f"({len(valid)} found, need ≥3).")

print(f"\n{'='*65}")
print(f"  Day 22B complete. Output → day22b_order_param_refined.png")
print(f"{'='*65}\n")