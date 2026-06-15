import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# DAY 21 — PHYSICS LAYER, DAY 1
# M/M/1 QUEUING THEORY MODEL
#
# GOAL: Can pure MATH predict where our
# system collapses — WITHOUT looking at
# the empirical cliff we already found?
#
# Day 20 found: collapse starts around batch=192
# (empirical = "we measured it and saw it happen")
#
# Today we ask: does queuing theory PREDICT
# a collapse near batch=192 too?
# If yes → our cliff isn't random. It follows
# a known physical law (this is the research claim).
# ─────────────────────────────────────────

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_FILE  = "/Users/devansh/Desktop/optimizeation/telemetry_v2.csv"
OUTPUT_PNG = "/Users/devansh/Desktop/optimizeation/day21_mm1_queuing.png"

SLA_THRESHOLD_MS = 20.0

# Traffic intensity threshold (rho_critical).
# In queuing theory, systems are considered
# "near breaking point" once rho > 0.85
# (this is a commonly used rule of thumb in
# call-center and network design — once a
# resource is 85% busy, wait times start
# growing very fast)
RHO_CRITICAL = 0.85

# When rho gets very close to 1, the formula
# W = rho / (1 - rho) blows up toward infinity
# (literally — dividing by a number near zero).
# We "clip" (cap) rho at this value so our
# math doesn't produce infinity/garbage numbers.
RHO_CLIP_MAX = 0.97

# ─────────────────────────────────────────
# STEP 1 — LOAD & CLEAN DATA
# (same cleaning approach as Day 19/20)
# ─────────────────────────────────────────
df = pd.read_csv(DATA_FILE)

df["cpu_usage"] = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage"] = (
    df["cpu_usage"]
    .rolling(window=5, min_periods=1, center=True)
    .median()
    .ffill().bfill()
)

df["avg_latency_ms"] = df["avg_latency"] * 1000
df["p95_latency_ms"] = df["p95_latency"] * 1000

# ─────────────────────────────────────────
# STEP 2 — AGGREGATE PER BATCH SIZE
# (one row per batch size, same as Day 20)
# ─────────────────────────────────────────
agg = df.groupby("batch_size").agg(
    avg_latency_ms = ("avg_latency_ms", "mean"),
    p95_latency_ms = ("p95_latency_ms", "mean"),
    throughput     = ("throughput",     "mean"),
).reset_index().sort_values("batch_size").reset_index(drop=True)

print("✅ Step 1-2 — Data loaded and aggregated\n")

# ─────────────────────────────────────────
# STEP 3 — COMPUTE TRAFFIC INTENSITY ρ(b)
#
# WHAT IS TRAFFIC INTENSITY?
# Think of your system like a single checkout
# counter. "Traffic intensity" (rho, written ρ)
# is a number between 0 and 1 that tells you
# how "busy" or "loaded" that counter is.
#
#   ρ = 0   → counter is empty, totally idle
#   ρ = 0.5 → counter is half-busy
#   ρ = 1   → counter is at 100% capacity,
#             completely overwhelmed
#
# HOW WE CALCULATE IT FROM OUR DATA:
# Your system's BEST throughput (most requests
# handled per second) happens at the smallest
# batch size — that's your system running
# closest to "ideal speed" (call this throughput_peak).
#
# As batch size grows and throughput drops,
# the system is doing WORSE relative to its
# best possible speed. We turn that into ρ:
#
#   ρ(b) = 1 - [ throughput(b) / throughput_peak ]
#
# If throughput(b) == throughput_peak → ρ = 0 (idle/healthy)
# If throughput(b) drops to half of peak → ρ = 0.5
# If throughput(b) drops close to 0 → ρ → 1 (overwhelmed)
# ─────────────────────────────────────────

throughput_peak = agg["throughput"].max()
peak_batch      = agg.loc[agg["throughput"].idxmax(), "batch_size"]

agg["rho"] = 1 - (agg["throughput"] / throughput_peak)

# Clip rho so it never goes negative (tiny
# measurement noise could push it just below 0)
# and never goes above RHO_CLIP_MAX (so our
# next formula doesn't explode to infinity)
agg["rho_clipped"] = agg["rho"].clip(lower=0.0, upper=RHO_CLIP_MAX)

print(f"✅ Step 3 — Traffic intensity ρ(b) computed")
print(f"   Peak throughput = {throughput_peak:.2f} "
      f"(at batch={int(peak_batch)}) → this is our ρ=0 reference")
print(f"   ρ range in data : "
      f"{agg['rho'].min():.3f} to {agg['rho'].max():.3f}\n")

# ─────────────────────────────────────────
# STEP 4 — COMPUTE THEORETICAL DELAY W(b)
#
# THE M/M/1 FORMULA (the famous one):
#   W = ρ / (1 - ρ)
#
# This formula says: as ρ gets closer to 1,
# the bottom part (1 - ρ) gets closer to 0,
# and dividing by a tiny number makes the
# whole thing EXPLODE upward.
#
# Example:
#   ρ = 0.5  → W = 0.5 / 0.5  = 1.0
#   ρ = 0.8  → W = 0.8 / 0.2  = 4.0
#   ρ = 0.9  → W = 0.9 / 0.1  = 9.0
#   ρ = 0.95 → W = 0.95/0.05  = 19.0
#   ρ = 0.99 → W = 0.99/0.01  = 99.0
#
# See how it's not a straight line? Going from
# 0.5 → 0.8 (+0.3) adds 3.0 to W. But going from
# 0.9 → 0.95 (+0.05, much smaller step!) adds
# 10.0 to W. This is the "cliff" shape — small
# changes near ρ=1 cause HUGE jumps in delay.
#
# This W(b) is a THEORETICAL number — it has no
# real "ms" unit yet. It's a SHAPE. We compare
# this shape to our REAL latency shape next.
# ─────────────────────────────────────────

agg["W_theory"] = agg["rho_clipped"] / (1 - agg["rho_clipped"])

print(f"✅ Step 4 — Theoretical delay W(b) computed "
      f"(unitless, shape only)\n")

# ─────────────────────────────────────────
# STEP 5 — NORMALIZE FOR COMPARISON
#
# WHAT DOES "NORMALIZE" MEAN?
# Our real latency is measured in milliseconds
# (e.g. 7ms to 21ms). Our theoretical W is a
# unitless number (e.g. 0 to 32). We can't
# compare 7ms to 0.5 directly — different scales!
#
# To compare SHAPES (not exact values), we
# rescale both curves to sit between 0 and 1.
# This is called MIN-MAX NORMALIZATION:
#
#   normalized_value = (value - min) / (max - min)
#
# After this, both curves start at 0 (their
# lowest point) and end at 1 (their highest
# point) — so we can lay them on top of each
# other and see if they follow the same SHAPE,
# even though their original units are different.
# ─────────────────────────────────────────

def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

agg["latency_norm"]  = normalize(agg["avg_latency_ms"])
agg["W_theory_norm"] = normalize(agg["W_theory"])

print(f"✅ Step 5 — Both curves normalized to 0-1 range "
      f"for shape comparison\n")

# ─────────────────────────────────────────
# STEP 6 — MODEL VALIDATION
# (does theory actually MATCH reality?)
#
# We use PEARSON CORRELATION (r) — the same
# tool from Day 8 — but this time we're not
# correlating two MEASUREMENTS. We're
# correlating a THEORETICAL PREDICTION with
# a REAL MEASUREMENT.
#
# r close to +1.0 → theory predicts reality
#                   very well (shapes match)
# r close to 0    → theory and reality are
#                   unrelated (model is wrong)
# ─────────────────────────────────────────

r_value = agg["W_theory"].corr(agg["avg_latency_ms"])

print(f"✅ Step 6 — Model validation correlation")
print(f"   r (W_theory vs actual latency) = {r_value:+.3f}\n")

# ─────────────────────────────────────────
# STEP 7 — FIND THE PREDICTED CRITICAL POINT
#
# We want to find: at what batch size does
# ρ(b) FIRST CROSS 0.85 (our RHO_CRITICAL)?
#
# Our data only has 13 batch sizes (8, 16, 32...512)
# — ρ doesn't land EXACTLY on 0.85 at any of them.
# So we use INTERPOLATION (a way to estimate a
# value BETWEEN two known points, by drawing a
# straight line between them and reading off
# where it crosses our target).
#
# np.interp(target_y, known_x, known_y) does
# this for us automatically.
# ─────────────────────────────────────────

batch_sizes = agg["batch_size"].values
rho_values  = agg["rho"].values

if rho_values.max() >= RHO_CRITICAL and rho_values.min() < RHO_CRITICAL:
    # np.interp needs the x-values (rho) to be increasing,
    # so we make sure they are sorted that way
    predicted_critical_batch = np.interp(
        RHO_CRITICAL, rho_values, batch_sizes
    )
else:
    predicted_critical_batch = np.nan

# Empirical critical point — from Day 20's findings.
# This is the batch size where the engine FIRST
# flagged "throughput_collapse" / HIGH severity.
empirical_critical_batch = 192

print(f"✅ Step 7 — Critical point comparison")
if not np.isnan(predicted_critical_batch):
    print(f"   Theory predicts  ρ crosses {RHO_CRITICAL} "
          f"at batch ≈ {predicted_critical_batch:.0f}")
else:
    print(f"   ρ never reaches {RHO_CRITICAL} in this dataset")
print(f"   Data shows collapse starting at batch = "
      f"{empirical_critical_batch}\n")

# ─────────────────────────────────────────
# STEP 8 — PLOT (4 panels)
# ─────────────────────────────────────────

colors = {
    "rho"      : "#534AB7",   # purple — traffic intensity
    "theory"   : "#D85A30",   # red-orange — theory curve
    "actual"   : "#1D9E75",   # green — real measured latency
    "threshold": "#E24B4A",   # red — critical threshold lines
    "gray"     : "#999999",
}

fig, axes = plt.subplots(4, 1, figsize=(13, 18))
fig.suptitle(
    "Day 21 — M/M/1 Queuing Theory Model (Physics Layer)\n"
    "System Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

# ── PANEL 1 — Traffic Intensity ρ(b) ──
#
# Shows how "loaded" the system is at each
# batch size. The horizontal red line is our
# RHO_CRITICAL = 0.85 threshold. Where the
# purple line crosses that red line is the
# THEORY'S prediction of where things go wrong.
ax1 = axes[0]

ax1.plot(
    agg["batch_size"], agg["rho"],
    color=colors["rho"], linewidth=2.5,
    marker="o", markersize=6,
    label="Traffic intensity ρ(b)"
)
ax1.axhline(
    y=RHO_CRITICAL, color=colors["threshold"],
    linewidth=2, linestyle="--",
    label=f"Critical threshold (ρ = {RHO_CRITICAL})"
)

if not np.isnan(predicted_critical_batch):
    ax1.axvline(
        x=predicted_critical_batch,
        color=colors["theory"], linewidth=2,
        linestyle=":",
        label=f"Theory predicts collapse at "
              f"batch ≈ {predicted_critical_batch:.0f}"
    )

ax1.axvline(
    x=empirical_critical_batch,
    color=colors["actual"], linewidth=2,
    linestyle=":",
    label=f"Data shows collapse at batch = "
          f"{empirical_critical_batch}"
)

ax1.set_title(
    "Traffic Intensity ρ(b) — How 'Loaded' Is The System?\n"
    "(ρ near 1 = system overwhelmed, theory's danger zone)",
    fontsize=11, fontweight="bold"
)
ax1.set_xlabel("Batch size")
ax1.set_ylabel("ρ (traffic intensity)")
ax1.set_xticks(batch_sizes)
ax1.set_ylim(-0.05, 1.05)
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(True, alpha=0.3)

# ── PANEL 2 — Theory vs Reality (normalized) ──
#
# Both curves rescaled to 0-1. If the SHAPES
# overlap closely, theory explains reality.
ax2 = axes[1]

ax2.plot(
    agg["batch_size"], agg["W_theory_norm"],
    color=colors["theory"], linewidth=2.5,
    marker="s", markersize=6,
    label="Theory: W(b) — predicted delay shape"
)
ax2.plot(
    agg["batch_size"], agg["latency_norm"],
    color=colors["actual"], linewidth=2.5,
    marker="o", markersize=6,
    label="Reality: actual latency(b) — measured"
)

ax2.set_title(
    "Theory vs Reality — Normalized Shape Comparison\n"
    "(both curves rescaled 0→1, do the shapes follow each other?)",
    fontsize=11, fontweight="bold"
)
ax2.set_xlabel("Batch size")
ax2.set_ylabel("Normalized value (0-1)")
ax2.set_xticks(batch_sizes)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── PANEL 3 — Validation scatter ──
#
# Each dot = one batch size.
# x = theory's predicted W(b)
# y = actual measured latency(b)
# If dots roughly fall on a line going up-right,
# theory and reality move together.
ax3 = axes[2]

ax3.scatter(
    agg["W_theory"], agg["avg_latency_ms"],
    color=colors["rho"], s=80, alpha=0.7,
    edgecolors="black", linewidth=0.5,
    zorder=5
)

# Label each point with its batch size
for _, row in agg.iterrows():
    ax3.annotate(
        f"  {int(row['batch_size'])}",
        (row["W_theory"], row["avg_latency_ms"]),
        fontsize=8
    )

# Best-fit line (a straight line that best
# follows the trend of the dots — drawn using
# numpy's polyfit, degree 1 = straight line)
z = np.polyfit(agg["W_theory"], agg["avg_latency_ms"], 1)
trend_x = np.linspace(agg["W_theory"].min(), agg["W_theory"].max(), 50)
trend_y = z[0] * trend_x + z[1]
ax3.plot(
    trend_x, trend_y,
    color=colors["threshold"], linewidth=2,
    linestyle="--", label=f"Best-fit line (r = {r_value:+.3f})"
)

ax3.set_title(
    "Model Validation — Theory's Prediction vs Actual Latency\n"
    "(each dot = one batch size, label = batch size)",
    fontsize=11, fontweight="bold"
)
ax3.set_xlabel("W_theory(b) — theoretical delay (unitless)")
ax3.set_ylabel("Actual avg latency (ms)")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ── PANEL 4 — Critical point comparison bar ──
#
# A simple side-by-side bar comparing where
# theory PREDICTS the collapse vs where the
# DATA actually showed it (Day 20's finding).
ax4 = axes[3]

labels = ["Theory Prediction\n(ρ crosses 0.85)",
          "Empirical Finding\n(Day 20 data)"]
values = [
    predicted_critical_batch if not np.isnan(predicted_critical_batch) else 0,
    empirical_critical_batch
]
bar_colors4 = [colors["theory"], colors["actual"]]

bars4 = ax4.bar(labels, values, color=bar_colors4, alpha=0.85, width=0.5)

for bar, val in zip(bars4, values):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        f"batch ≈ {val:.0f}",
        ha="center", va="bottom",
        fontsize=11, fontweight="bold"
    )

if not np.isnan(predicted_critical_batch):
    diff = abs(predicted_critical_batch - empirical_critical_batch)
    diff_pct = (diff / empirical_critical_batch) * 100
    ax4.text(
        0.5, max(values) * 1.15,
        f"Difference: {diff:.0f} "
        f"({diff_pct:.0f}% of empirical value)",
        ha="center", transform=ax4.get_xaxis_transform(),
        fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

ax4.set_title(
    "Critical Point — Theory vs Data\n"
    "(does pure math land near where we actually measured collapse?)",
    fontsize=11, fontweight="bold"
)
ax4.set_ylabel("Batch size")
ax4.set_ylim(0, max(values) * 1.35)
ax4.grid(True, alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"✅ Chart saved → {OUTPUT_PNG}\n")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("=" * 65)
print("  DAY 21 FINDINGS — M/M/1 QUEUING THEORY MODEL")
print("=" * 65)

print(f"\n── Traffic Intensity ρ(b) Table ──\n")
print(f"  {'Batch':<8} {'Throughput':<12} {'ρ(b)':<8} "
      f"{'W_theory':<10} {'Actual Latency'}")
print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*10} {'-'*15}")
for _, row in agg.iterrows():
    flag = " ⚠️" if row["rho"] >= RHO_CRITICAL else ""
    print(f"  {int(row['batch_size']):<8} "
          f"{row['throughput']:<12.2f} "
          f"{row['rho']:<8.3f} "
          f"{row['W_theory']:<10.2f} "
          f"{row['avg_latency_ms']:.2f}ms{flag}")

print(f"\n── Model Validation ──\n")
print(f"  Correlation (theory vs actual): r = {r_value:+.3f}")
if abs(r_value) > 0.8:
    verdict = "STRONG — theory explains the cliff very well"
elif abs(r_value) > 0.5:
    verdict = "MODERATE — theory partially explains the cliff"
else:
    verdict = "WEAK — theory does not explain this cliff well, " \
              "other factors (CPU/memory/ECE layer) likely dominate"
print(f"  Verdict: {verdict}")

print(f"\n── Critical Point Comparison ──\n")
print(f"  Theory predicts collapse near : "
      f"batch ≈ {predicted_critical_batch:.0f}" 
      if not np.isnan(predicted_critical_batch)
      else "  Theory: ρ never reached critical threshold in this range")
print(f"  Data shows collapse starting  : batch = {empirical_critical_batch}")

if not np.isnan(predicted_critical_batch):
    diff = abs(predicted_critical_batch - empirical_critical_batch)
    diff_pct = (diff / empirical_critical_batch) * 100
    print(f"  Difference                    : {diff:.0f} "
          f"({diff_pct:.0f}%)")
    if diff_pct < 25:
        print(f"\n  ✅ CLOSE MATCH — queuing theory independently "
              f"predicts a collapse")
        print(f"     point within {diff_pct:.0f}% of what we "
              f"empirically measured.")
        print(f"     This supports the claim: the cliff at "
              f"batch={empirical_critical_batch} is")
        print(f"     not arbitrary — it follows known queuing "
              f"theory behaviour.")
    else:
        print(f"\n  ⚠️  GAP — theory and data disagree by "
              f"{diff_pct:.0f}%.")
        print(f"     This suggests the collapse is driven by "
              f"something queuing")
        print(f"     theory alone doesn't capture (e.g. memory "
              f"bandwidth limits —")
        print(f"     this is exactly what the ECE/Roofline layer "
              f"investigates next).")

print(f"\n{'='*65}")
print(f"  Day 21 complete. Output → day21_mm1_queuing.png")
print(f"{'='*65}\n")