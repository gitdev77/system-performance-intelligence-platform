### Layer 1 — Data Collection
- **Workload:** Matrix multiply A(batch×512) × B(512×512), float32, Apple Silicon
- **Telemetry:** psutil captures CPU, memory, latency per run
- **Dataset:** 104 rows, 13 batch sizes [8,16,32,...,512], 8 runs each
- **Key columns:** avg_latency (seconds), p95_latency, throughput, cpu_usage

### Layer 2 — Physics & Analytics
- **M/M/1 Queuing Theory:** Models system as single-server queue. 
  Theoretical delay W(b) correlates r=0.974 with actual latency.
- **Phase Transition Analysis:** Order parameter Φ(b) detects critical point b_c=32.
- **ECE Roofline Model:** Arithmetic intensity AI(b) classifies workload as 
  memory-bound. Rules out memory bandwidth as cliff cause.
- **Bottleneck Engine:** KMeans K=3 clusters identify Safe/Warning/Danger regimes.
- **Anomaly Detection:** Z-score + IQR flags statistical outliers per batch size.

### Layer 3 — ML Prediction
- **Features:** batch_size, throughput, cpu_usage, cv_latency, rho (M/M/1), AI (Roofline)
- **Model:** Logistic Regression, physics-informed features
- **Performance:** precision=1.0, recall=1.0, AUC=1.0 at threshold=0.7
- **Key insight:** Physics feature rho (utilization) is strongest predictor (weight=+1.532)
- **Experiment tracking:** 9 MLflow runs across Model A/B/C variants

### Layer 4 — API & MLOps
- **FastAPI:** 5 endpoints — /health, /predict, /diagnose, /recommend, /cost/{batch_size}
- **Docker:** Containerized as "system-perf-api", clean requirements.txt
- **PostgreSQL:** 2 tables (telemetry_runs 104 rows, batch_summary 13 rows)
- **JSON Logging:** JSONL format, 4 log levels, session summaries
- **Drift Detection:** Model robust to 16.7% perturbation, FPR < 1.1%

### Layer 5 — Observability & Presentation
- **Prometheus:** Scrapes /metrics at port 8001 every 15 seconds
- **Grafana:** 4 live panels — latency, p95 latency, SLA breaches, batch size gauge
- **Tableau:** 4-chart dashboard — latency trend, CPU saturation, SLA heatmap, cost analysis
- **Cost Model:** batch=8 costs $0.000008/request vs batch=512 costs $0.050512/request

---

## 3. Key Findings

| Finding | Value |
|---------|-------|
| Throughput-latency correlation | r = -0.982 |
| M/M/1 queuing validation | r = 0.974 |
| Performance cliff location | batch = 192 (37.5% SLA breach rate) |
| ML model precision | 1.0 |
| ML model recall | 1.0 |
| ML model AUC | 1.0 |
| Cost cliff ratio | batch=512 is 6,314× more expensive than batch=8 |
| Annual saving (1M daily requests) | $356,441 |
| Recommended operating point | batch=8 or batch=16 |
| Safety margin from danger zone | 120 batch units |

---

## 4. Design Decisions

### Why Logistic Regression over Neural Networks?
Dataset has 104 rows — far too small for neural networks. Logistic Regression is 
interpretable, fast, and achieves perfect scores. Physics features provide the 
non-linearity that the model itself doesn't need to learn.

### Why M/M/1 Queuing Theory?
The inference server behaves like a single-server queue — one GPU/CPU processes 
requests sequentially. M/M/1 gives us a theoretical baseline that r=0.974 confirms 
matches reality. This validates our physical understanding of the system.

### Why p95 latency for SLA (not average)?
Average latency hides tail pain. At batch=512, avg latency = 20.8ms (barely over 
threshold) but p95 = 28.6ms (43% over threshold). P95 represents the worst 5% of 
real user experiences — that's what SLAs should protect.

### Why Docker for the API?
Reproducibility. Any machine can run `docker-compose up` and get the exact same 
environment. No "works on my machine" problems.

### Why Prometheus + Grafana over just logging?
Logs answer "what happened?" Prometheus + Grafana answers "what is happening RIGHT NOW?" 
Production ML systems need both. The live SLA breach counter (533 breaches tracked 
in real-time) demonstrates the system works under realistic load.

---

## 5. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Workload | NumPy | Matrix multiply simulation |
| Monitoring | psutil | System telemetry collection |
| Analytics | Pandas, SciPy | Data processing, statistics |
| Physics | Custom Python | M/M/1, Roofline, Phase Transition |
| ML | Scikit-learn | Logistic Regression, KMeans |
| Experiment Tracking | MLflow | 9 runs logged, model registry |
| API | FastAPI + Uvicorn | REST endpoints, Swagger UI |
| Containerization | Docker | Reproducible deployment |
| Database | PostgreSQL + psycopg2 | Persistent analytics storage |
| Monitoring | Prometheus + Grafana | Real-time observability |
| Visualization | Tableau, Matplotlib | Dashboards, static charts |
| Logging | JSONL | Structured audit trail |

---

## 6. Limitations

- Single-node synthetic workload — no network I/O, no distributed inference
- Apple Silicon specific behavior — results may differ on x86 or GPU clusters
- 104 row dataset — perfect ML scores may reflect dataset simplicity
- Batch sizes tested: 8–512 — behavior beyond 512 unknown
- Cost model uses simplified penalty function — real SLA penalties are contractual

---

## 7. Future Work (Post Day 44)

- Extend to GPU inference workloads (CUDA, MPS)
- Add distributed tracing (OpenTelemetry)
- Online learning — model updates as new telemetry arrives
- Multi-node testing to validate M/M/1 assumptions at scale
- LLM-powered natural language alerts (Days 41-43)