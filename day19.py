import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# DAY 19 — CPU SPIKE ISOLATION ENGINE
#
# GOAL: Find out if CPU spikes are REAL
# latency drivers, or just measurement noise.
#
# WHY THIS MATTERS:
# Day 8 told us CPU-latency correlation = +0.37 (weak).
# But Pearson (a formula that measures average
# straight-line relationship) averages everything.
# If CPU only spikes in 2 out of 8 runs, those
# spikes get drowned out by the other 6 clean runs.
# We need to SEPARATE spiked runs from clean runs
# and compare them directly.
# ─────────────────────────────────────────

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_FILE       = "/Users/devansh/Desktop/optimizeation/telemetry_v2.csv"
OUTPUT_PNG      = "/Users/devansh/Desktop/optimizeation/day19_cpu_spike.png"
SLA_THRESHOLD_MS = 20.0

# CPU spike threshold:
# Any run where cpu_usage > this is a "spike run"
# We use 75% as the threshold because your data shows
# CPU normally sits around 55-65%. Anything above 75%
# is meaningfully elevated — not just noise.
CPU_SPIKE_THRESHOLD = 75.0

# ─────────────────────────────────────────
# STEP 1 — LOAD & CLEAN DATA
# ─────────────────────────────────────────
df = pd.read_csv(DATA_FILE)

# Convert latency columns to milliseconds
# (original data is in seconds, ms is easier to read)
df["avg_latency_ms"] = df["avg_latency"] * 1000
df["p95_latency_ms"] = df["p95_latency"] * 1000
df["p99_latency_ms"] = df["p99_latency"] * 1000

# ── CLEAN CPU SIGNAL ──
#
# Problem: psutil sometimes returns 0.0 on Apple Silicon
# when the polling interval is too short. That's not real.
#
# Old fix: .ffill() — just copy the previous valid value.
# That's lazy and can propagate wrong values for many rows.
#
# Better fix: Rolling Median Filter
# (rolling = look at a sliding window of N rows at a time,
#  median = pick the middle value, which ignores extremes)
#
# We use a window of 5 rows. For each row, look at the
# 2 rows before it, itself, and 2 rows after it.
# Take the median of those 5 cpu values.
# If the original value is 0.0, replace with the median.
# This preserves real high-CPU readings while killing zeros.

df["cpu_rolling_median"] = (
    df["cpu_usage"]
    .replace(0.0, np.nan)          # treat 0.0 as missing
    .rolling(window=5,
             min_periods=1,        # (min_periods=1 means:
             center=True)          #  even at the edges of
    .median()                      #  the data, still compute)
)

# Fill any remaining NaN (at edges) with forward fill
df["cpu_rolling_median"] = df["cpu_rolling_median"].ffill().bfill()

# Use cleaned CPU for all analysis
df["cpu_clean"] = np.where(
    df["cpu_usage"] == 0.0,
    df["cpu_rolling_median"],
    df["cpu_usage"]
)

print("✅ Step 1 — Data loaded and CPU signal cleaned")
print(f"   Rows with cpu=0.0 (bad readings): "
      f"{(df['cpu_usage'] == 0.0).sum()}")
print(f"   These were replaced with rolling median\n")

# ─────────────────────────────────────────
# STEP 2 — LABEL EACH RUN AS SPIKE OR CLEAN
#
# This is called "conditional labeling" —
# we split the data into two groups based on
# a condition (cpu > threshold), then compare
# the two groups on some other metric (latency).
# ─────────────────────────────────────────
df["is_spike"] = df["cpu_clean"] > CPU_SPIKE_THRESHOLD

spike_runs = df[df["is_spike"] == True]
clean_runs = df[df["is_spike"] == False]

total      = len(df)
n_spike    = len(spike_runs)
n_clean    = len(clean_runs)
spike_pct  = (n_spike / total) * 100

print(f"✅ Step 2 — Spike labeling done")
print(f"   CPU spike threshold : {CPU_SPIKE_THRESHOLD}%")
print(f"   Spike runs          : {n_spike}/{total} ({spike_pct:.1f}%)")
print(f"   Clean runs          : {n_clean}/{total} "
      f"({100-spike_pct:.1f}%)\n")

# ─────────────────────────────────────────
# STEP 3 — BUILD SPIKE ATTRIBUTION TABLE
#
# For each batch size, compute:
#  - how many runs were spike runs
#  - avg latency of spike runs
#  - avg latency of clean runs
#  - latency delta (spike_avg - clean_avg)
#
# "Attribution" = assigning blame. We're asking:
#  "at batch=128, how much extra latency can we
#   attribute to CPU spikes specifically?"
# ─────────────────────────────────────────
batch_sizes = sorted(df["batch_size"].unique())
attribution = []

for batch in batch_sizes:
    batch_df    = df[df["batch_size"] == batch]
    spike_df    = batch_df[batch_df["is_spike"] == True]
    clean_df    = batch_df[batch_df["is_spike"] == False]

    n_total     = len(batch_df)
    n_spk       = len(spike_df)
    spike_freq  = (n_spk / n_total) * 100 if n_total > 0 else 0

    avg_spike_lat = spike_df["avg_latency_ms"].mean() \
                    if len(spike_df) > 0 else np.nan
    avg_clean_lat = clean_df["avg_latency_ms"].mean() \
                    if len(clean_df) > 0 else np.nan

    # Delta: how much worse is latency during spike runs?
    # Positive delta = spike runs are slower (bad)
    # Negative delta = spike runs are faster (weird, means
    #                  CPU isn't driving latency here)
    delta = avg_spike_lat - avg_clean_lat \
            if not (np.isnan(avg_spike_lat) or
                    np.isnan(avg_clean_lat)) \
            else np.nan

    # Severity score (0-1 scale):
    # combines spike frequency and latency delta
    # High score = CPU spikes are both frequent AND causing pain
    severity = (spike_freq / 100) * max(0, delta) \
               if not np.isnan(delta) else 0

    attribution.append({
        "batch_size"    : batch,
        "n_total"       : n_total,
        "n_spike"       : n_spk,
        "spike_freq_pct": round(spike_freq, 1),
        "avg_spike_lat" : round(avg_spike_lat, 2)
                          if not np.isnan(avg_spike_lat) else np.nan,
        "avg_clean_lat" : round(avg_clean_lat, 2)
                          if not np.isnan(avg_clean_lat) else np.nan,
        "delta_ms"      : round(delta, 2)
                          if not np.isnan(delta) else np.nan,
        "severity_score": round(severity, 3),
    })

attr_df = pd.DataFrame(attribution)

print("✅ Step 3 — Spike attribution table built")
print(f"   Batch sizes analyzed: {len(attr_df)}\n")

# ─────────────────────────────────────────
# STEP 4 — GLOBAL SPIKE vs CLEAN COMPARISON
#
# Compute overall stats across ALL batch sizes.
# This gives a single headline number for the findings.
# ─────────────────────────────────────────
overall_spike_avg = spike_runs["avg_latency_ms"].mean()
overall_clean_avg = clean_runs["avg_latency_ms"].mean()
overall_delta     = overall_spike_avg - overall_clean_avg
overall_pct_worse = (overall_delta / overall_clean_avg) * 100

# Pearson correlation using CLEANED cpu signal
# (recompute to see if cleaning changes the story)
corr_raw    = df["cpu_usage"].corr(df["avg_latency_ms"])
corr_clean  = df["cpu_clean"].corr(df["avg_latency_ms"])

print("✅ Step 4 — Global comparison computed")
print(f"   Spike run avg latency : {overall_spike_avg:.2f}ms")
print(f"   Clean run avg latency : {overall_clean_avg:.2f}ms")
print(f"   Delta                 : +{overall_delta:.2f}ms "
      f"({overall_pct_worse:+.1f}%)")
print(f"   CPU-latency r (raw)   : {corr_raw:+.3f}")
print(f"   CPU-latency r (clean) : {corr_clean:+.3f}\n")

# ─────────────────────────────────────────
# STEP 5 — PLOT (4 panels)
# ─────────────────────────────────────────
colors = {
    "spike"   : "#D85A30",   # red-orange for spike runs
    "clean"   : "#1D9E75",   # green for clean runs
    "sla"     : "#E24B4A",   # red for SLA line
    "neutral" : "#534AB7",   # purple for neutral signals
    "warning" : "#EF9F27",   # orange for warnings
    "gray"    : "#999999",
}

fig, axes = plt.subplots(4, 1, figsize=(13, 20))
fig.suptitle(
    "Day 19 — CPU Spike Isolation Engine\n"
    "System Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

batch_arr = attr_df["batch_size"].values

# ── PANEL 1 — Raw CPU scatter across all runs ──
#
# A scatter plot shows every individual data point
# as a dot. This lets you SEE the distribution —
# where does CPU cluster? Where are the outliers?
ax1 = axes[0]

# Separate spike and clean indices for coloring
spike_idx = df[df["is_spike"] == True].index
clean_idx = df[df["is_spike"] == False].index

ax1.scatter(
    df.loc[clean_idx, "batch_size"],
    df.loc[clean_idx, "cpu_clean"],
    color=colors["clean"], alpha=0.5, s=25,
    label=f"Clean run (cpu ≤ {CPU_SPIKE_THRESHOLD}%)"
)
ax1.scatter(
    df.loc[spike_idx, "batch_size"],
    df.loc[spike_idx, "cpu_clean"],
    color=colors["spike"], alpha=0.7, s=40,
    label=f"Spike run (cpu > {CPU_SPIKE_THRESHOLD}%)",
    zorder=5   # (zorder = drawing layer; higher = on top)
)

# Draw the spike threshold line
ax1.axhline(
    y=CPU_SPIKE_THRESHOLD,
    color=colors["spike"], linewidth=2,
    linestyle="--",
    label=f"Spike threshold ({CPU_SPIKE_THRESHOLD}%)"
)

ax1.set_title(
    "CPU Usage — Every Individual Run\n"
    f"(red dots = spike runs where cpu > {CPU_SPIKE_THRESHOLD}%)",
    fontsize=11, fontweight="bold"
)
ax1.set_ylabel("CPU Usage (%)")
ax1.set_xlabel("Batch size")
ax1.set_xticks(batch_sizes)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 110)

# ── PANEL 2 — Latency: spike runs vs clean runs ──
#
# Box plots are perfect here.
# A box plot shows you:
#   - the median (middle line in the box)
#   - the middle 50% of values (the box itself,
#     called the IQR = Interquartile Range)
#   - the min and max (the whiskers)
#   - outliers (dots outside the whiskers)
# Side-by-side boxes make comparison easy.
ax2 = axes[1]

# Build data arrays for boxplot — one list per batch size
spike_lat_by_batch = []
clean_lat_by_batch = []
x_positions_spike  = []
x_positions_clean  = []

# We need numeric x positions for the box plots
# Map batch sizes to index positions
batch_to_x = {b: i for i, b in enumerate(batch_sizes)}

for batch in batch_sizes:
    bdf = df[df["batch_size"] == batch]
    s_lat = bdf[bdf["is_spike"] == True]["avg_latency_ms"].values
    c_lat = bdf[bdf["is_spike"] == False]["avg_latency_ms"].values

    x = batch_to_x[batch]
    if len(s_lat) > 0:
        spike_lat_by_batch.append(s_lat)
        x_positions_spike.append(x - 0.2)
    if len(c_lat) > 0:
        clean_lat_by_batch.append(c_lat)
        x_positions_clean.append(x + 0.2)

# Draw the box plots
bp_spike = ax2.boxplot(
    spike_lat_by_batch,
    positions=x_positions_spike,
    widths=0.35,
    patch_artist=True,    # (patch_artist=True makes
                          #  filled colored boxes,
                          #  not just outlines)
    boxprops=dict(facecolor=colors["spike"], alpha=0.6),
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(color=colors["spike"]),
    capprops=dict(color=colors["spike"]),
    flierprops=dict(marker="o", color=colors["spike"],
                    markersize=4, alpha=0.5)
)
bp_clean = ax2.boxplot(
    clean_lat_by_batch,
    positions=x_positions_clean,
    widths=0.35,
    patch_artist=True,
    boxprops=dict(facecolor=colors["clean"], alpha=0.6),
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(color=colors["clean"]),
    capprops=dict(color=colors["clean"]),
    flierprops=dict(marker="o", color=colors["clean"],
                    markersize=4, alpha=0.5)
)

ax2.axhline(
    y=SLA_THRESHOLD_MS,
    color=colors["sla"], linewidth=1.5,
    linestyle="-.", label="SLA threshold (20ms)"
)

# Custom x-tick labels
ax2.set_xticks(range(len(batch_sizes)))
ax2.set_xticklabels(batch_sizes)

# Legend patches (since boxplot doesn't auto-generate legend)
legend_patch_spike = mpatches.Patch(
    color=colors["spike"], alpha=0.7, label="Spike runs"
)
legend_patch_clean = mpatches.Patch(
    color=colors["clean"], alpha=0.7, label="Clean runs"
)
ax2.legend(
    handles=[legend_patch_spike, legend_patch_clean],
    fontsize=8
)

ax2.set_title(
    "Latency Distribution — Spike Runs vs Clean Runs\n"
    "(do CPU spikes actually cause higher latency?)",
    fontsize=11, fontweight="bold"
)
ax2.set_ylabel("Avg Latency (ms)")
ax2.set_xlabel("Batch size")
ax2.grid(True, alpha=0.3, axis="y")

# ── PANEL 3 — Delta bar chart ──
#
# For each batch size, show the latency penalty
# caused by CPU spikes.
# delta_ms = spike_avg - clean_avg
# Positive = spike runs were slower (CPU caused pain)
# Negative = spike runs were actually faster
#            (CPU isn't the driver at this batch size)
ax3 = axes[2]

valid = attr_df.dropna(subset=["delta_ms"])

bar_colors3 = [
    colors["spike"] if d > 0 else colors["clean"]
    for d in valid["delta_ms"]
]

bars3 = ax3.bar(
    valid["batch_size"],
    valid["delta_ms"],
    color=bar_colors3,
    alpha=0.8,
    width=20
)

# Zero reference line
ax3.axhline(y=0, color="black", linewidth=1)

# Label each bar with its delta value
for bar, val in zip(bars3, valid["delta_ms"]):
    ypos = bar.get_height() + 0.3 if val >= 0 \
           else bar.get_height() - 1.2
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        ypos,
        f"{val:+.1f}ms",
        ha="center", va="bottom",
        fontsize=8, fontweight="bold"
    )

ax3.set_title(
    "Latency Penalty From CPU Spikes — Per Batch Size\n"
    "(positive = spike runs slower | negative = spike runs faster)",
    fontsize=11, fontweight="bold"
)
ax3.set_ylabel("Latency Delta (ms)\nSpike − Clean")
ax3.set_xlabel("Batch size")
ax3.set_xticks(valid["batch_size"])
ax3.grid(True, alpha=0.3, axis="y")

# ── PANEL 4 — Spike frequency heatmap ──
#
# A heatmap uses color to encode a number.
# Here: red = high spike frequency at that batch size
#       green = low spike frequency
# This instantly shows WHERE spikes are concentrated.
ax4 = axes[3]

# Build 2-row heatmap matrix:
# Row 0 = spike frequency (%)
# Row 1 = severity score (frequency × latency delta)
heatmap_data = np.array([
    attr_df["spike_freq_pct"].values,
    attr_df["severity_score"].fillna(0).values * 10
    # (×10 just to normalize severity to a similar
    #  scale as frequency % for visual comparison)
])

im = ax4.imshow(
    heatmap_data,
    cmap="RdYlGn_r",   # (RdYlGn_r = Red-Yellow-Green reversed
                        #  so high values = red = bad)
    aspect="auto",
    vmin=0, vmax=100
)

plt.colorbar(im, ax=ax4, shrink=0.6, label="Score (0-100)")

ax4.set_xticks(range(len(batch_sizes)))
ax4.set_xticklabels(batch_sizes, fontsize=8)
ax4.set_yticks([0, 1])
ax4.set_yticklabels([
    "Spike Frequency (%)",
    "Severity Score (×10)"
], fontsize=9)

# Annotate each cell with the number
for col_i, batch in enumerate(attr_df["batch_size"]):
    freq_val = heatmap_data[0, col_i]
    sev_val  = heatmap_data[1, col_i]
    ax4.text(col_i, 0, f"{freq_val:.0f}%",
             ha="center", va="center",
             fontsize=8, fontweight="bold",
             color="white" if freq_val > 50 else "black")
    ax4.text(col_i, 1, f"{sev_val:.1f}",
             ha="center", va="center",
             fontsize=8, fontweight="bold",
             color="white" if sev_val > 50 else "black")

ax4.set_title(
    "Spike Frequency & Severity Heatmap — Per Batch Size\n"
    "(dark red = high spike frequency + high latency impact)",
    fontsize=11, fontweight="bold"
)
ax4.set_xlabel("Batch size")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"✅ Chart saved → {OUTPUT_PNG}\n")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("=" * 62)
print("  DAY 19 FINDINGS — CPU SPIKE ISOLATION ENGINE")
print("=" * 62)

print(f"\n── Global Stats ──\n")
print(f"  Total runs          : {total}")
print(f"  Spike runs          : {n_spike} ({spike_pct:.1f}%)")
print(f"  Clean runs          : {n_clean} ({100-spike_pct:.1f}%)")
print(f"  CPU threshold used  : {CPU_SPIKE_THRESHOLD}%")

print(f"\n── Latency Impact ──\n")
print(f"  Spike run avg latency  : {overall_spike_avg:.2f}ms")
print(f"  Clean run avg latency  : {overall_clean_avg:.2f}ms")
print(f"  Latency penalty        : +{overall_delta:.2f}ms "
      f"({overall_pct_worse:+.1f}%)")

print(f"\n── Correlation Update ──\n")
print(f"  CPU-latency r (raw zeros kept) : {corr_raw:+.3f}")
print(f"  CPU-latency r (cleaned signal) : {corr_clean:+.3f}")
delta_r = corr_clean - corr_raw
print(f"  Change after cleaning          : {delta_r:+.3f}")

print(f"\n── Per-Batch Spike Attribution ──\n")
print(f"  {'Batch':<8} {'Spikes':<10} {'Spike Lat':<12} "
      f"{'Clean Lat':<12} {'Delta':<10} {'Severity'}")
print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*8}")

for _, row in attr_df.iterrows():
    spike_lat_str = (f"{row['avg_spike_lat']:.2f}ms"
                     if not np.isnan(row["avg_spike_lat"])
                     else "  N/A  ")
    delta_str     = (f"{row['delta_ms']:+.2f}ms"
                     if not np.isnan(row["delta_ms"])
                     else "  N/A  ")
    flag = " ⚠️" if (not np.isnan(row["delta_ms"])
                     and row["delta_ms"] > 2.0) else ""
    print(f"  {int(row['batch_size']):<8} "
          f"{row['n_spike']}/{row['n_total']} runs  "
          f"{spike_lat_str:<12} "
          f"{row['avg_clean_lat']:.2f}ms       "
          f"{delta_str:<10} "
          f"{row['severity_score']:.3f}{flag}")

# Verdict
print(f"\n── VERDICT ──\n")
significant = attr_df[
    attr_df["delta_ms"].notna() & (attr_df["delta_ms"] > 2.0)
]
if len(significant) > 0:
    bad_batches = significant["batch_size"].tolist()
    print(f"  ✅ CPU spikes DO cause real latency pain at:")
    print(f"     batch sizes = {bad_batches}")
    print(f"  → CPU spike frequency should be a FEATURE")
    print(f"    in the Day 22 ML model.")
else:
    print(f"  ✅ CPU spikes cause minimal latency impact (<2ms)")
    print(f"     across all batch sizes.")
    print(f"  → CPU spike count is LOW PRIORITY as an ML feature.")
    print(f"     Throughput and batch_size remain dominant drivers.")

print(f"\n  Overall: spike runs are {overall_pct_worse:+.1f}% "
      f"{'slower' if overall_pct_worse > 0 else 'faster'} "
      f"than clean runs.")
print(f"  Cleaned correlation ({corr_clean:+.3f}) vs "
      f"raw ({corr_raw:+.3f}) — "
      f"{'cleaning revealed stronger signal' if abs(corr_clean) > abs(corr_raw) else 'cleaning made minimal difference'}.")
print(f"\n{'='*62}")
print(f"  Day 19 complete. Output → day19_cpu_spike.png")
print(f"{'='*62}\n")