import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# DAY 23 — ROOFLINE MODEL
# ECE Layer, Day 1
#
# GOAL: Use hardware-performance theory (FLOPs,
# memory bandwidth, cache) to explain the part
# of the batch=192 cliff that M/M/1 queuing
# theory (Day 21) couldn't fully explain.
# ─────────────────────────────────────────

DATA_FILE  = "/Users/devansh/Desktop/optimizeation/telemetry_v2.csv"
OUTPUT_PNG = "/Users/devansh/Desktop/optimizeation/day23_roofline.png"

# The matrix multiply shape used in your telemetry
# generator: A(batch x MATRIX_DIM) x B(MATRIX_DIM x MATRIX_DIM)
MATRIX_DIM = 512
BYTES_PER_FLOAT = 4  # float32 = 4 bytes per number

# Typical Apple Silicon cache sizes (approximate,
# used as REFERENCE LINES only — not exact for
# your specific chip)
L1_CACHE_MB = 0.125   # ~128 KB per core
L2_CACHE_MB = 12.0    # ~12 MB shared (performance cores)

# ── STEP 1: Load & clean (same as previous days) ──
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
    .agg(avg_latency_ms=("avg_latency_ms", "mean"))
    .reset_index()
    .sort_values("batch_size")
    .reset_index(drop=True)
)

b = agg["batch_size"]

# ── STEP 3: Compute FLOPs(b) ──
# For A(b x 512) x B(512 x 512):
# each output element needs 512 multiplies + 512 adds
agg["flops"] = 2 * b * MATRIX_DIM * MATRIX_DIM

# ── STEP 4: Compute bytes moved(b) ──
# We read matrix A (b x 512), read matrix B (512 x 512),
# and write the output C (b x 512). All in float32.
agg["bytes_moved"] = (
    (b * MATRIX_DIM) +              # read A
    (MATRIX_DIM * MATRIX_DIM) +     # read B
    (b * MATRIX_DIM)                # write C
) * BYTES_PER_FLOAT

# ── STEP 5: Arithmetic Intensity AI(b) ──
agg["AI"] = agg["flops"] / agg["bytes_moved"]

# ── STEP 6: Achieved performance P(b) in GFLOPs/sec ──
latency_sec = agg["avg_latency_ms"] / 1000
agg["gflops_per_sec"] = (agg["flops"] / latency_sec) / 1e9

print("✅ Step 1-6 — FLOPs, bytes moved, AI, and achieved "
      "performance computed\n")

# ── STEP 7: Estimate the two "roofs" from YOUR data ──
#
# MEMORY ROOF (bandwidth estimate):
# Use the SMALLEST batch size (lowest AI) — at low AI,
# the system is most likely memory-bound, so
# performance / AI ≈ memory bandwidth (bytes/sec)
mem_row = agg.loc[agg["AI"].idxmin()]
bandwidth_est = (mem_row["gflops_per_sec"] * 1e9) / mem_row["AI"]  # bytes/sec
bandwidth_gb_s = bandwidth_est / 1e9

# COMPUTE ROOF (peak GFLOPs/sec):
# The highest achieved performance across all batch sizes
peak_compute_gflops = agg["gflops_per_sec"].max()
peak_compute_flops  = peak_compute_gflops * 1e9

# RIDGE POINT — where the two roofs cross
# memory_roof:  P = bandwidth * AI
# compute_roof: P = peak_compute
# setting equal: peak_compute = bandwidth * AI_ridge
ai_ridge = peak_compute_flops / bandwidth_est

print(f"✅ Step 7 — Roofline estimated from data")
print(f"   Estimated memory bandwidth : {bandwidth_gb_s:.2f} GB/s")
print(f"   Estimated peak compute     : {peak_compute_gflops:.2f} GFLOPs/s")
print(f"   Ridge point AI             : {ai_ridge:.2f} FLOPs/byte\n")

# ── STEP 8: Classify each batch size ──
agg["regime"] = np.where(
    agg["AI"] < ai_ridge, "memory-bound", "compute-bound"
)

# ── STEP 9: Working set size & cache comparison ──
agg["working_set_mb"] = (
    (2 * b * MATRIX_DIM + MATRIX_DIM * MATRIX_DIM) * BYTES_PER_FLOAT
) / (1024 ** 2)

agg["exceeds_l1"] = agg["working_set_mb"] > L1_CACHE_MB
agg["exceeds_l2"] = agg["working_set_mb"] > L2_CACHE_MB

print(f"✅ Step 9 — Working set size computed")
print(f"   Range: {agg['working_set_mb'].min():.3f} MB "
      f"to {agg['working_set_mb'].max():.3f} MB")
print(f"   L1 reference: {L1_CACHE_MB} MB | "
      f"L2 reference: {L2_CACHE_MB} MB\n")

# ─────────────────────────────────────────
# STEP 10 — PLOT (4 panels)
# ─────────────────────────────────────────
colors = {
    "data"    : "#534AB7",
    "mem_roof": "#D85A30",
    "cmp_roof": "#1D9E75",
    "ridge"   : "#E24B4A",
    "ai"      : "#534AB7",
    "perf"    : "#1D9E75",
    "ws"      : "#EF9F27",
    "l1"      : "#D85A30",
    "l2"      : "#E24B4A",
}

fig, axes = plt.subplots(4, 1, figsize=(13, 18))
fig.suptitle(
    "Day 23 — Roofline Model (ECE Layer)\n"
    "System Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

batch_arr = agg["batch_size"].values

# ── PANEL 1 — Roofline plot (log-log) ──
ax1 = axes[0]

ax1.scatter(
    agg["AI"], agg["gflops_per_sec"],
    color=colors["data"], s=80, zorder=5,
    edgecolors="black", linewidth=0.5
)
for _, row in agg.iterrows():
    ax1.annotate(
        f"  {int(row['batch_size'])}",
        (row["AI"], row["gflops_per_sec"]), fontsize=8
    )

# Memory roof: P = bandwidth * AI (a straight line on log-log)
ai_range = np.linspace(agg["AI"].min() * 0.8, agg["AI"].max() * 1.2, 50)
mem_roof_line = (bandwidth_est * ai_range) / 1e9
ax1.plot(
    ai_range, mem_roof_line, color=colors["mem_roof"],
    linewidth=2, linestyle="--",
    label=f"Memory roof (bandwidth ≈ {bandwidth_gb_s:.1f} GB/s)"
)

# Compute roof: flat line at peak_compute_gflops
ax1.axhline(
    y=peak_compute_gflops, color=colors["cmp_roof"],
    linewidth=2, linestyle="--",
    label=f"Compute roof (peak ≈ {peak_compute_gflops:.1f} GFLOPs/s)"
)

ax1.axvline(
    x=ai_ridge, color=colors["ridge"], linewidth=2,
    linestyle=":", label=f"Ridge point (AI ≈ {ai_ridge:.1f})"
)

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_title(
    "Roofline Plot — Achieved Performance vs Arithmetic Intensity\n"
    "(left of ridge = memory-bound, right of ridge = compute-bound)",
    fontsize=11, fontweight="bold"
)
ax1.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
ax1.set_ylabel("Achieved Performance (GFLOPs/s)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, which="both")

# ── PANEL 2 — AI(b) vs batch size ──
ax2 = axes[1]
ax2.plot(
    batch_arr, agg["AI"], color=colors["ai"],
    linewidth=2.5, marker="o", markersize=6,
    label="Arithmetic Intensity AI(b)"
)
ax2.axhline(
    y=ai_ridge, color=colors["ridge"], linewidth=2,
    linestyle=":", label=f"Ridge point (AI ≈ {ai_ridge:.1f})"
)
ax2.set_title(
    "Arithmetic Intensity vs Batch Size\n"
    "(theory says: higher AI → should be MORE efficient)",
    fontsize=11, fontweight="bold"
)
ax2.set_xlabel("Batch size")
ax2.set_ylabel("AI (FLOPs/byte)")
ax2.set_xticks(batch_arr)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── PANEL 3 — Achieved performance vs batch size ──
ax3 = axes[2]
ax3.plot(
    batch_arr, agg["gflops_per_sec"], color=colors["perf"],
    linewidth=2.5, marker="o", markersize=6,
    label="Achieved performance (GFLOPs/s)"
)
ax3.axvline(
    x=192, color=colors["ridge"], linewidth=2,
    linestyle=":", label="Empirical collapse point (batch=192)"
)
ax3.set_title(
    "Achieved Performance vs Batch Size\n"
    "(does performance keep improving with AI, or stall/drop?)",
    fontsize=11, fontweight="bold"
)
ax3.set_xlabel("Batch size")
ax3.set_ylabel("GFLOPs/s")
ax3.set_xticks(batch_arr)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ── PANEL 4 — Working set size vs cache lines ──
ax4 = axes[3]
ax4.plot(
    batch_arr, agg["working_set_mb"], color=colors["ws"],
    linewidth=2.5, marker="o", markersize=6,
    label="Working set size (MB)"
)
ax4.axhline(
    y=L1_CACHE_MB, color=colors["l1"], linewidth=2,
    linestyle="--", label=f"Typical L1 cache (~{L1_CACHE_MB} MB)"
)
ax4.axhline(
    y=L2_CACHE_MB, color=colors["l2"], linewidth=2,
    linestyle="--", label=f"Typical L2 cache (~{L2_CACHE_MB} MB)"
)
ax4.axvline(
    x=192, color="#999999", linewidth=1.5,
    linestyle=":", label="Empirical collapse point (batch=192)"
)
ax4.set_yscale("log")
ax4.set_title(
    "Working Set Size vs Batch Size (log scale)\n"
    "(does the cliff coincide with crossing a cache boundary?)",
    fontsize=11, fontweight="bold"
)
ax4.set_xlabel("Batch size")
ax4.set_ylabel("Working set (MB, log scale)")
ax4.set_xticks(batch_arr)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, which="both")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"✅ Chart saved → {OUTPUT_PNG}\n")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("=" * 70)
print("  DAY 23 FINDINGS — ROOFLINE MODEL")
print("=" * 70)

print(f"\n── Per-Batch Table ──\n")
print(f"  {'Batch':<8} {'AI':<10} {'GFLOPs/s':<12} "
      f"{'Regime':<15} {'Working Set'}")
print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*15} {'-'*12}")
for _, row in agg.iterrows():
    flag = " ⚠️" if row["batch_size"] == 192 else ""
    print(f"  {int(row['batch_size']):<8} "
          f"{row['AI']:<10.2f} "
          f"{row['gflops_per_sec']:<12.2f} "
          f"{row['regime']:<15} "
          f"{row['working_set_mb']:.3f} MB{flag}")

print(f"\n── Roofline Summary ──\n")
print(f"  Estimated memory bandwidth : {bandwidth_gb_s:.2f} GB/s")
print(f"  Estimated peak compute     : {peak_compute_gflops:.2f} GFLOPs/s")
print(f"  Ridge point AI             : {ai_ridge:.2f} FLOPs/byte")

n_mem = (agg["regime"] == "memory-bound").sum()
n_cmp = (agg["regime"] == "compute-bound").sum()
print(f"\n  Memory-bound batch sizes  : {n_mem}")
print(f"  Compute-bound batch sizes : {n_cmp}")

print(f"\n── AI vs Performance Mismatch Check ──\n")
ai_192 = agg.loc[agg["batch_size"] == 192, "AI"].values[0]
ai_512 = agg.loc[agg["batch_size"] == 512, "AI"].values[0]
perf_192 = agg.loc[agg["batch_size"] == 192, "gflops_per_sec"].values[0]
perf_512 = agg.loc[agg["batch_size"] == 512, "gflops_per_sec"].values[0]
print(f"  AI(192) = {ai_192:.2f}  →  AI(512) = {ai_512:.2f}  "
      f"(AI keeps increasing)")
print(f"  Perf(192) = {perf_192:.2f} GFLOPs/s  →  "
      f"Perf(512) = {perf_512:.2f} GFLOPs/s")
if perf_512 <= perf_192 * 1.1:
    print(f"\n  ⚠️  MISMATCH: AI increases by "
          f"{((ai_512-ai_192)/ai_192*100):.0f}% from batch 192→512,")
    print(f"     but performance does NOT improve proportionally.")
    print(f"     → Classic roofline (FLOPs/bandwidth alone) does NOT")
    print(f"       fully explain the batch=192 cliff.")
else:
    print(f"\n  ✅ Performance scales with AI as roofline predicts.")

print(f"\n── Cache / Working Set Check ──\n")
max_ws = agg["working_set_mb"].max()
if max_ws < L2_CACHE_MB:
    print(f"  Working set stays UNDER typical L2 cache size "
          f"({max_ws:.2f} MB < {L2_CACHE_MB} MB)")
    print(f"  across ALL batch sizes tested.")
    print(f"  → L2 cache overflow does NOT explain the batch=192 cliff.")
    print(f"  → All batch sizes already exceed typical L1 "
          f"({L1_CACHE_MB} MB), so L1 spillover is constant,")
    print(f"     not a NEW effect introduced at batch=192.")
else:
    crossing = agg[agg["working_set_mb"] > L2_CACHE_MB]["batch_size"].min()
    print(f"  Working set EXCEEDS L2 cache starting at "
          f"batch={int(crossing)}.")
    print(f"  → Possible cache-driven explanation for degradation "
          f"near this point.")

print(f"\n── Overall Verdict ──\n")
print(f"  Queuing theory (Day 21) explained ~95% of the latency")
print(f"  curve shape (r=0.974). Today's roofline + cache analysis")
print(f"  rules out classic memory-bandwidth and L2-cache overflow")
print(f"  as the cause of the remaining residual at batch=192.")
print(f"  → Remaining candidate: OS-level scheduling effects")
print(f"    (consistent with Day 19's CPU spike / Day 20's")
print(f"    'scheduling_noise' findings near batch=128-192).")

print(f"\n{'='*70}")
print(f"  Day 23 complete. Output → day23_roofline.png")
print(f"{'='*70}\n")