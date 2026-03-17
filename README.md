# System Performance Intelligence Platform
> 🚧 Active Development — Day 8/60 | Building in public, one day at a time.

## What This Is
A production-grade performance analytics engine that:
- Simulates ML inference workloads under varying batch sizes
- Captures real telemetry (CPU, memory, latency, throughput)
- Automatically detects performance cliffs using derivative analysis
- Identifies system bottlenecks through statistical correlation
- (Coming) Predicts SLA breaches using ML
- (Coming) Models cost-per-request tradeoffs

## Key Findings So Far
- Throughput collapse (r=-0.888) is the dominant latency driver
- CPU is NOT the bottleneck (r=+0.368)
- Performance cliff detected at batch=192 (inflection point)
- First SLA breach at batch=512 (p95 > 20ms)

## Tech Stack
Python, NumPy, Pandas, Matplotlib, Scikit-learn (coming), SQL (coming)

## Daily Progress
- [x] Day 1-6: Telemetry engine + dataset generation (104 rows, 13 columns)
- [x] Day 7: Automatic cliff detection engine + 3-panel visualization
- [x] Day 8: Correlation matrix + scatter analysis + bottleneck identification
- [ ] Day 9-14: Performance analytics layer
- [ ] Day 15-21: Bottleneck intelligence engine
- [ ] Day 22-28: ML prediction layer (SLA breach prediction)
- [ ] Day 29-35: Monitoring simulation layer
- [ ] Day 36-42: Cost modeling layer
- [ ] Day 43-49: Dashboard (SQL + Power BI/Tableau)
- [ ] Day 50-60: Polish + documentation + GitHub finalization

## Project Structure
optimization/
├── code8.py          # Telemetry generator
├── cliff.py          # Cliff detection engine
├── correlation8.py   # Correlation analysis
├── scatter8.py       # Scatter analysis
├── telemetry_v2.csv  # Raw telemetry dataset
└── cliff_analysis.csv # Processed cliff data

Updated Day 9 analysis