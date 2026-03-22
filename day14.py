import pandas as pd
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("/Users/devansh/Desktop/optimizeation/telemetry_v2.csv")

df["cpu_usage"]     = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage"]     = df["cpu_usage"].ffill()
df["avg_latency_ms"] = df["avg_latency"] * 1000
df["p95_latency_ms"] = df["p95_latency"] * 1000
df["p99_latency_ms"] = df["p99_latency"] * 1000

SLA_THRESHOLD_MS    = 20.0

# ─────────────────────────────────────────
# COMPUTE ALL SUMMARY STATS
# ─────────────────────────────────────────
agg = df.groupby("batch_size").agg(
    avg_latency_ms  = ("avg_latency_ms",  "mean"),
    p95_latency_ms  = ("p95_latency_ms",  "mean"),
    p99_latency_ms  = ("p99_latency_ms",  "mean"),
    cpu_usage       = ("cpu_usage",       "mean"),
    throughput      = ("throughput",      "mean"),
    cv_latency      = ("cv_latency",      "mean"),
).reset_index()

# SLA breach per batch
df["sla_breach_v2"] = (df["p95_latency_ms"] > SLA_THRESHOLD_MS).astype(int)
breach_by_batch = df.groupby("batch_size")["sla_breach_v2"].mean() * 100

# Key numbers
total_runs        = len(df)
total_batches     = df["batch_size"].nunique()
overall_breach    = df["sla_breach_v2"].mean() * 100
peak_throughput   = agg["throughput"].max()
peak_tput_batch   = agg.loc[agg["throughput"].idxmax(), "batch_size"]
min_latency       = agg["avg_latency_ms"].min()
min_lat_batch     = agg.loc[agg["avg_latency_ms"].idxmin(), "batch_size"]
max_latency       = agg["avg_latency_ms"].max()
max_lat_batch     = agg.loc[agg["avg_latency_ms"].idxmax(), "batch_size"]
safe_batches      = breach_by_batch[breach_by_batch == 0].index.tolist()
critical_batches  = breach_by_batch[breach_by_batch >= 50].index.tolist()

# Correlation with avg latency
corr = df[[
    "batch_size", "avg_latency_ms",
    "cpu_usage", "memory_usage", "throughput"
]].corr()["avg_latency_ms"]

# ─────────────────────────────────────────
# GENERATE REPORT
# ─────────────────────────────────────────
report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║         WEEK 2 PERFORMANCE ANALYSIS REPORT                         ║
║         System Performance Intelligence Platform                    ║
║         Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}                      ║
╚══════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. EXPERIMENT OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Platform      : Apple MacBook Air (Apple Silicon M-series)
  Workload      : Simulated ML inference (matrix multiplication)
  Dataset       : {total_runs} experiments across {total_batches} batch sizes
  Batch range   : 8 → 512 (13 operating points)
  Repeats       : 8 runs per batch size
  SLA threshold : p95 latency < {SLA_THRESHOLD_MS}ms

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. KEY FINDINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  FINDING 1 — Latency scales 2.9x across batch range
  ─────────────────────────────────────────────────────
  Minimum avg latency : {min_latency:.2f}ms at batch={int(min_lat_batch)}
  Maximum avg latency : {max_latency:.2f}ms at batch={int(max_lat_batch)}
  Growth factor       : {max_latency/min_latency:.2f}x
  Pattern             : Gradual growth, no single catastrophic cliff

  FINDING 2 — Throughput peaks at small batch sizes
  ─────────────────────────────────────────────────────
  Peak throughput     : {peak_throughput:.1f} req/s at batch={int(peak_tput_batch)}
  Throughput at 512   : {agg.loc[agg['batch_size']==512,'throughput'].values[0]:.1f} req/s
  Collapse factor     : {peak_throughput/agg.loc[agg['batch_size']==512,'throughput'].values[0]:.2f}x drop from peak to batch=512
  Implication         : Larger batches process more data but serve
                        fewer requests per second

  FINDING 3 — CPU is NOT the bottleneck
  ─────────────────────────────────────────────────────
  CPU vs latency correlation  : r={corr['cpu_usage']:+.3f} (WEAK)
  Batch vs latency correlation: r={corr['batch_size']:+.3f} (MODERATE)
  Throughput vs latency       : r={corr['throughput']:+.3f} (STRONG)
  Conclusion: Computation scaling drives latency, not CPU saturation.
              Apple Silicon handles CPU load efficiently across all
              batch sizes (~55-65% CPU throughout).

  FINDING 4 — SLA breach pattern
  ─────────────────────────────────────────────────────
  Safe batch sizes    : {safe_batches} → 0% breach rate
  Critical batches    : {critical_batches} → 50%+ breach rate
  Overall breach rate : {overall_breach:.1f}% (SLA allows 0.1%)
  Error budget status : EXCEEDED by {overall_breach/0.1:.0f}x

  FINDING 5 — Two distinct degradation patterns
  ─────────────────────────────────────────────────────
  Pattern A (Sustained) : batch=512 — consistently high latency
                          avg=20.9ms, p99=30.5ms, predictable
  Pattern B (Spikes)    : batch=128, 320 — random latency spikes
                          p99/p50 ratio 2.5x, 2.6x — unpredictable
  Implication           : Spike pattern harder to detect and fix
                          than sustained high latency

  FINDING 6 — Tail latency inequality
  ─────────────────────────────────────────────────────
  Best consistency    : batch=448 (p50/p99 ratio = 0.64)
  Worst consistency   : batch=320 (p50/p99 ratio = 0.39)
  No batch size achieved Good consistency (>0.7)
  Implication         : System never fully predictable —
                        worst users always get 1.6x-2.6x
                        worse experience than typical users

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. BOTTLENECK ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Primary bottleneck  : Computation scaling with batch size
  Secondary factor    : Non-deterministic OS scheduling causing
                        latency spikes at moderate batch sizes
  NOT a bottleneck    : CPU utilization (r=+0.37, weak)
  NOT a bottleneck    : Memory usage (r=+0.39, spurious correlation)

  Root cause          : Matrix multiplication complexity grows as
                        O(n²) with batch size — each doubling of
                        batch size more than doubles computation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. CAPACITY PLANNING RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  RECOMMENDATION 1 — Safe production batch size
  Maximum recommended batch size: 64
  Rationale: Zero SLA breaches below batch=64. Beyond this point
             the system enters Warning zone with increasing
             breach probability.

  RECOMMENDATION 2 — Avoid spike-prone batch sizes
  Avoid batch=128 and batch=320 under sustained load.
  Rationale: These show worst p99/p50 ratios (2.5x, 2.6x)
             indicating high variance — unpredictable for users.

  RECOMMENDATION 3 — Horizontal scaling trigger point
  Scale horizontally when batch size demand exceeds 64.
  Rationale: Beyond batch=64 it is more cost-effective to add
             another inference instance than to absorb the
             latency and SLA breach cost.

  RECOMMENDATION 4 — Monitoring alert thresholds
  Alert on p95 > 15ms  (early warning — 75% of SLA limit)
  Alert on p95 > 20ms  (SLA breach — immediate action needed)
  Alert on p99 > 25ms  (tail spike — investigate scheduling)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. WEEK 2 DELIVERABLES COMPLETED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Day 8  ✅  Correlation matrix — bottleneck identification
  Day 9  ✅  Latency vs CPU dual axis analysis
  Day 10 ✅  Non-linear growth detection engine
  Day 11 ✅  Performance Risk Index dashboard
  Day 12 ✅  SLA breach frequency + error budget analysis
  Day 13 ✅  Tail latency metrics — p50, p90, p95, p99
  Day 14 ✅  Mini performance analysis report (this document)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. NEXT PHASE — WEEK 3 (BOTTLENECK INTELLIGENCE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Day 15-16  Multi-variable bottleneck analysis
  Day 17     Artificial memory pressure simulation
  Day 18     Light vs heavy load regime comparison
  Day 19     Performance cliff precise identification
  Day 20     Bottleneck detection logic engine
  Day 21     Bottleneck classification system

╔══════════════════════════════════════════════════════════════════════╗
║  END OF REPORT                                                      ║
║  System Performance Intelligence Platform — Week 2 Complete        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────
# PRINT AND SAVE REPORT
# ─────────────────────────────────────────
print(report)

report_path = "/Users/devansh/Desktop/optimizeation/week2_performance_report.txt"
with open(report_path, "w") as f:
    f.write(report)

print(f"✅ Report saved → week2_performance_report.txt")