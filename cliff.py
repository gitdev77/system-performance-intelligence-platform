import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────
df = pd.read_csv("telemetry_v2.csv")

# ─────────────────────────────────────────
# STEP 1 — Aggregate by batch size
# ─────────────────────────────────────────
agg = df.groupby("batch_size").agg(
    avg_latency   = ("avg_latency",  "mean"),
    p95_latency   = ("p95_latency",  "mean"),
    p99_latency   = ("p99_latency",  "mean"),
    cpu_usage     = ("cpu_usage",    "mean"),
    memory_usage  = ("memory_usage", "mean"),
    cv_latency    = ("cv_latency",   "mean"),
    throughput    = ("throughput",   "mean"),
).reset_index()

# ─────────────────────────────────────────
# STEP 2 — Compute latency derivative
# Rate of change of avg_latency between batch sizes
# This is the core of cliff detection
# ─────────────────────────────────────────
agg["latency_delta"]      = agg["avg_latency"].diff()
agg["batch_delta"]        = agg["batch_size"].diff()
agg["latency_derivative"] = agg["latency_delta"] / agg["batch_delta"]

# ─────────────────────────────────────────
# STEP 3 — Detect cliff
# Cliff = derivative spikes above threshold
# Threshold = mean + 1.5 * std of derivative
# ─────────────────────────────────────────
deriv_mean = agg["latency_derivative"].mean()
deriv_std  = agg["latency_derivative"].std()
cliff_threshold = deriv_mean + 1.5 * deriv_std

agg["cliff_flag"] = (agg["latency_derivative"] > cliff_threshold).astype(int)

# ─────────────────────────────────────────
# STEP 4 — Identify saturation zone
# Everything after first cliff flag = saturation
# ─────────────────────────────────────────
first_cliff_idx = agg[agg["cliff_flag"] == 1].index.min()

agg["saturation_zone"] = 0
if not pd.isna(first_cliff_idx):
    agg.loc[first_cliff_idx:, "saturation_zone"] = 1

# ─────────────────────────────────────────
# STEP 5 — Fix SLA threshold
# Your data's p95 max is ~25ms so tighten SLA
# ─────────────────────────────────────────
SLA_THRESHOLD = 0.020  # 20ms — realistic for your data
agg["sla_breach"] = (agg["p95_latency"] > SLA_THRESHOLD).astype(int)

# ─────────────────────────────────────────
# STEP 6 — Rolling p99 drift detection
# Flag when p99 increases >20% over previous batch
# ─────────────────────────────────────────
agg["p99_drift"] = agg["p99_latency"].pct_change()
agg["p99_drift_flag"] = (agg["p99_drift"] > 0.20).astype(int)

# ─────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────
print("=" * 70)
print("PERFORMANCE CLIFF DETECTION REPORT")
print("=" * 70)

print(f"\nDerivative threshold (cliff trigger): {cliff_threshold:.6f}")
print(f"Derivative mean: {deriv_mean:.6f} | std: {deriv_std:.6f}")

print("\n── Aggregated metrics per batch size ──\n")
print(agg[[
    "batch_size",
    "avg_latency",
    "p95_latency",
    "latency_derivative",
    "cliff_flag",
    "saturation_zone",
    "sla_breach",
    "p99_drift_flag"
]].to_string(index=False))

# Cliff summary
cliff_batches = agg[agg["cliff_flag"] == 1]["batch_size"].tolist()
saturation_start = agg[agg["saturation_zone"] == 1]["batch_size"].min()
sla_breach_batches = agg[agg["sla_breach"] == 1]["batch_size"].tolist()

print("\n" + "=" * 70)
print("FINDINGS")
print("=" * 70)
print(f"Cliff detected at batch sizes : {cliff_batches}")
print(f"Saturation zone starts at     : batch={saturation_start}")
print(f"SLA breaches at batch sizes   : {sla_breach_batches}")
print(f"p99 drift spikes at           : {agg[agg['p99_drift_flag']==1]['batch_size'].tolist()}")

# Save enriched dataset
agg.to_csv("cliff_analysis.csv", index=False)
print("\n✅ Saved → cliff_analysis.csv")