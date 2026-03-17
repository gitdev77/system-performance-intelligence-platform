# System Performance Intelligence Platform
> 🚧 Active Development — Day 9/60 | Building in public, one day at a time.

## What This Is
A production-grade performance analytics engine that:
- Simulates ML inference workloads under varying batch sizes
- Captures real telemetry (CPU, memory, latency, throughput)
- Automatically detects performance cliffs using derivative analysis
- Identifies system bottlenecks through statistical correlation
- Plots dual-axis latency vs CPU behavior across load regimes
- (Coming) Predicts SLA breaches using ML
- (Coming) Models cost-per-request tradeoffs
- (Coming) Full monitoring simulation + dashboard

## Key Findings So Far
- Throughput collapse (r=-0.888) is the dominant latency driver
- CPU is NOT the bottleneck (r=+0.368) — Apple Silicon handles load efficiently
- Performance cliff inflection point at batch=192
- p95 tail latency breaches 20ms SLA from batch=128 onwards
- Avg latency only breaches SLA at batch=512 — avg hides tail pain
- CPU stays flat at ~60% while latency grows 2.6x — computation scaling is root cause

## Tech Stack
Python, NumPy, Pandas, Matplotlib, Scikit-learn (coming), SQL (coming), Power BI (coming)

## Daily Progress
- [x] Day 1: Python + psutil setup, basic CPU/memory monitoring script
- [x] Day 2: Heavy workload simulation (matrix multiplication), latency measurement
- [x] Day 3: Multiple repetitions, random batch sizes, structured CSV dataset
- [x] Day 4: Throughput calculation, SLA threshold, SLA breach flag
- [x] Day 5: 100+ experiments, large dataset generation, data cleaning
- [x] Day 6: Basic EDA (mean, max, distribution), latency vs batch size plot
- [x] Day 7: Tail latency (p95, p99), automatic cliff detection engine, 3-panel visualization
- [x] Day 8: Pearson correlation matrix, scatter analysis, bottleneck identification
- [x] Day 9: Dual-axis latency vs CPU chart, rolling average smoothing, crossover detection
- [ ] Day 10: Latency curve vs CPU usage deep dive
- [ ] Day 11: Non-linear growth region detection, saturation threshold
- [ ] Day 12: SLA breach frequency analysis
- [ ] Day 13: Tail latency metrics (p50, p90, p95, p99)
- [ ] Day 14: Mini performance analysis report
- [ ] Day 15-21: Bottleneck intelligence engine
- [ ] Day 22-28: ML prediction layer (SLA breach prediction)
- [ ] Day 29-35: Monitoring simulation layer
- [ ] Day 36-42: Cost modeling layer
- [ ] Day 43-49: Dashboard (SQL + Power BI/Tableau)
- [ ] Day 50-60: Polish + documentation + GitHub finalization


## Research Framing
**Hypothesis:** CPU saturation above 80% causes super-linear latency growth.

**Finding:** Rejected. CPU remained stable at ~60% across all batch sizes.
True cause: computation scaling with batch size (matrix multiply complexity).

**Limitations:** Single-node, synthetic workload, no network I/O, Apple Silicon specific behavior.



Updated Day 9 analysis

Email fix test commit