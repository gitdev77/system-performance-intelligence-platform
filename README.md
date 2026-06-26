


 A production-grade ML system that predicts SLA breaches, detects performance cliffs, and optimizes batch size selection for ML inference workloads — saving **$356,441/year** at 1M daily requests.

---
## ⚡ Key Findings (TL;DR for Recruiters)

| Finding | Value |
|---------|-------|
| Throughput-latency correlation | r = -0.982 |
| M/M/1 queuing theory validation | r = 0.974 |
| Performance cliff location | batch = 192 |
| ML model precision / recall | 1.0 / 1.0 |
| Optimal threshold (zero false negatives) | 0.7 |
| Cost cliff multiplier | 6,314× (batch=8 vs batch=512) |
| Annual saving from optimal batch selection | $356,441 |
| Recommended operating point | batch = 8 or 16 |

---
## 🏗️ System Architecture

```text
Raw Telemetry (104 runs, 13 batch sizes)
│
▼
┌─────────────────────────────────────────────────┐
│  Physics Layer                                  │
│  ├── M/M/1 Queuing Theory  (r=0.974)            │
│  ├── Phase Transition Analysis (b_c=32)         │
│  └── ECE Roofline Model (memory-bound)          │
└─────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────┐
│  ML Layer                                       │
│  ├── Logistic Regression (physics-informed)     │
│  ├── Perfect precision=1.0, recall=1.0          │
│  └── MLflow experiment tracking (9 runs)        │
└─────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────┐
│  Production Layer                               │
│  ├── FastAPI (5 endpoints)                      │
│  ├── Docker containerization                    │
│  ├── PostgreSQL (104 rows, 8 analytics queries) │
│  ├── Prometheus + Grafana (live monitoring)     │
│  └── Tableau dashboard (4 charts)               │
└─────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────┐
│  LLM Layer (Gemini 1.5 Flash)                   │
│  ├── Natural language performance reports       │
│  ├── Intelligent alert generation               │
│  └── Auto recommendation engine                 │
└─────────────────────────────────────────────────┘

🎯 Problem StatementML inference systems face a critical challenge: batch size selection directly controls the cost-performance tradeoff, yet most teams choose batch sizes arbitrarily.This project answers: At what batch size does the system cliff, why does it happen, and what is the optimal operating point?Workload: Matrix multiply A(batch×512) × B(512×512), float32, Apple Silicon MacDataset: 104 telemetry runs across 13 batch sizes [8→512], 8 runs each🔬 ApproachWeek 1-2: Telemetry FoundationBuilt psutil-based system monitor and workload simulatorGenerated 104-row dataset with 13 batch sizesKey discovery: throughput is primary latency driver (r=-0.982), CPU is NOT the bottleneck (r=-0.137)Week 3: Bottleneck IntelligenceRule-based bottleneck classifier + KMeans regime detection (K=3)Anomaly detection via Z-score + IQRPerformance cliff confirmed at batch=192Physics Layer (Days 22-24)M/M/1 Queuing Theory: theoretical delay W(b) correlates r=0.974 with actual latencyPhase Transition Analysis: critical point b_c=32 detectedECE Roofline Model: memory bandwidth ruled out as cliff causeML Layer (Days 25-29)Physics-informed features (rho from M/M/1, AI from Roofline)Logistic Regression: perfect precision=1.0, recall=1.0, AUC=1.0MLflow tracked 9 experiments across 3 model variants and 3 thresholdsOptimal threshold=0.7 gives zero false negatives across all sessionsCost Modeling (Days 30-31)Cost per request = compute cost + SLA penaltyPareto frontier analysis across 3 business scenariosAll scenarios agree: batch=8 recommended📊 ResultsPerformance CliffPlaintextBatch Size │ Avg Latency │ P95 Latency │ Breach Rate │ Regime
───────────┼─────────────┼─────────────┼─────────────┼─────────
8          │ 7.15ms      │ 9.61ms      │ 0%          │ Safe ✅
96         │ 10.16ms     │ 15.50ms     │ 12.5%       │ Safe ⚠️
128        │ 11.87ms     │ 21.36ms     │ 25%         │ Warning ⚠️
192        │ 18.50ms     │ 29.38ms     │ 50%         │ Danger 🔴
512        │ 20.85ms     │ 28.61ms     │ 100%        │ Danger 🔴
Cost AnalysisPlaintextBatch Size │ Cost/Request │ vs batch=8
───────────┼──────────────┼───────────
8          │ $0.000008    │ baseline
96         │ $0.006346    │ 793×
192        │ $0.025192    │ 3,149×
512        │ $0.050512    │ 6,314×
ML Model PerformancePlaintextModel          │ Precision │ Recall │ F1   │ AUC
───────────────┼───────────┼────────┼──────┼─────
Model A (raw)  │ 1.000     │ 1.000  │ 1.000│ 1.000
Model B (phys) │ 1.000     │ 1.000  │ 1.000│ 1.000
Best threshold │ 0.7       │ —      │ —    │ —
🛠️ Tech StackLayerTechnologyLanguagePython 3.12DataNumPy, PandasMLScikit-learn, MLflowPhysicsSciPy (M/M/1, Roofline)APIFastAPI, UvicornInfrastructureDocker, PostgreSQLMonitoringPrometheus, GrafanaVisualizationTableau, MatplotlibLLMGemini 1.5 Flash🚀 How to Run1. Clone the repoBashgit clone [https://github.com/gitdev77/system-performance-intelligence-platform](https://github.com/gitdev77/system-performance-intelligence-platform)
cd system-performance-intelligence-platform
2. Install dependenciesBashpip install -r requirements.txt
3. Run the FastAPI serverBashpython fastapiwrapper.py
# API live at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
4. Start monitoring stackBashpython day36_exporter.py &
docker-compose up -d
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
5. Launch MLflow UIBashmlflow ui
# MLflow: http://localhost:5000
📁 Project StructurePlaintext├── telemetry_v2.csv          # 104-row dataset
├── day24_train.csv           # ML train split (83 rows)
├── day24_test.csv            # ML test split (21 rows)
├── fastapiwrapper.py         # FastAPI — 5 endpoints
├── day36_exporter.py         # Prometheus metrics exporter
├── day37_mlflow.py           # MLflow experiment tracking
├── docker-compose.yml        # Prometheus + Grafana stack
├── prometheus.yml            # Prometheus scrape config
├── dockerfile                # Docker image definition
├── requirements.txt          # Python dependencies
└── SQL.py                    # PostgreSQL analytics layer
📈 Daily ProgressDayTopicStatus1-7Telemetry foundation✅8-14Performance analytics✅15-21Bottleneck intelligence✅22-24Physics layer✅25-26ML model✅27-29MLOps monitoring✅30-31Cost modeling✅32FastAPI✅33Docker✅34PostgreSQL✅35Tableau dashboard✅36Prometheus + Grafana✅37MLflow tracking✅38Architecture diagram✅39README✅40Pytest suite🔄41-43LLM Layer (Gemini)🔄44GitHub finalization🔄🔗 AuthorDevansh — Building in public, one day at a time.GitHub: @gitdev77Built with Python, physics, and persistence. 44 days of daily commits."""with open("README.md", "w") as f:f.write(readme)print("✅ README.md generated successfully!")print(f"   Total characters: {len(readme):,}")print(f"   Total lines: {readme.count(chr(10)):,}")print("\n   Preview of sections:")for line in readme.split("\n"):if line.startswith("## "):print(f"   {line}")
***

### 2. Clean Markdown Copy (For Direct Usage)

If you'd rather bypass the generator script and paste it directly into your GitHub repository, use the block below:

```markdown
# System Performance Intelligence & Cost-Aware Optimization Platform
> 🚀 A production-grade ML system that predicts SLA breaches, detects performance cliffs, and optimizes batch size selection for ML inference workloads — saving **$356,441/year** at 1M daily requests.

---
## ⚡ Key Findings (TL;DR for Recruiters)

| Finding | Value |
|---------|-------|
| Throughput-latency correlation | r = -0.982 |
| M/M/1 queuing theory validation | r = 0.974 |
| Performance cliff location | batch = 192 |
| ML model precision / recall | 1.0 / 1.0 |
| Optimal threshold (zero false negatives) | 0.7 |
| Cost cliff multiplier | 6,314× (batch=8 vs batch=512) |
| Annual saving from optimal batch selection | $356,441 |
| Recommended operating point | batch = 8 or 16 |

---
## 🏗️ System Architecture

```text
Raw Telemetry (104 runs, 13 batch sizes)
│
▼
┌─────────────────────────────────────────────────┐
│  Physics Layer                                  │
│  ├── M/M/1 Queuing Theory  (r=0.974)            │
│  ├── Phase Transition Analysis (b_c=32)         │
│  └── ECE Roofline Model (memory-bound)          │
└─────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────┐
│  ML Layer                                       │
│  ├── Logistic Regression (physics-informed)     │
│  ├── Perfect precision=1.0, recall=1.0          │
│  └── MLflow experiment tracking (9 runs)        │
└─────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────┐
│  Production Layer                               │
│  ├── FastAPI (5 endpoints)                      │
│  ├── Docker containerization                    │
│  ├── PostgreSQL (104 rows, 8 analytics queries) │
│  ├── Prometheus + Grafana (live monitoring)     │
│  └── Tableau dashboard (4 charts)               │
└─────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────┐
│  LLM Layer (Gemini 1.5 Flash)                   │
│  ├── Natural language performance reports       │
│  ├── Intelligent alert generation               │
│  └── Auto recommendation engine                 │
└─────────────────────────────────────────────────┘
🎯 Problem StatementML inference systems face a critical challenge: batch size selection directly controls the cost-performance tradeoff, yet most teams choose batch sizes arbitrarily.This project answers: At what batch size does the system cliff, why does it happen, and what is the optimal operating point?Workload: Matrix multiply A(batch×512) × B(512×512), float32, Apple Silicon MacDataset: 104 telemetry runs across 13 batch sizes [8→512], 8 runs each🔬 ApproachWeek 1-2: Telemetry FoundationBuilt psutil-based system monitor and workload simulatorGenerated 104-row dataset with 13 batch sizesKey discovery: throughput is primary latency driver (r=-0.982), CPU is NOT the bottleneck (r=-0.137)Week 3: Bottleneck IntelligenceRule-based bottleneck classifier + KMeans regime detection (K=3)Anomaly detection via Z-score + IQRPerformance cliff confirmed at batch=192Physics Layer (Days 22-24)M/M/1 Queuing Theory: theoretical delay W(b) correlates r=0.974 with actual latencyPhase Transition Analysis: critical point b_c=32 detectedECE Roofline Model: memory bandwidth ruled out as cliff causeML Layer (Days 25-29)Physics-informed features (rho from M/M/1, AI from Roofline)Logistic Regression: perfect precision=1.0, recall=1.0, AUC=1.0MLflow tracked 9 experiments across 3 model variants and 3 thresholdsOptimal threshold=0.7 gives zero false negatives across all sessionsCost Modeling (Days 30-31)Cost per request = compute cost + SLA penaltyPareto frontier analysis across 3 business scenariosAll scenarios agree: batch=8 recommended📊 ResultsPerformance CliffPlaintextBatch Size │ Avg Latency │ P95 Latency │ Breach Rate │ Regime
───────────┼─────────────┼─────────────┼─────────────┼─────────
8          │ 7.15ms      │ 9.61ms      │ 0%          │ Safe ✅
96         │ 10.16ms     │ 15.50ms     │ 12.5%       │ Safe ⚠️
128        │ 11.87ms     │ 21.36ms     │ 25%         │ Warning ⚠️
192        │ 18.50ms     │ 29.38ms     │ 50%         │ Danger 🔴
512        │ 20.85ms     │ 28.61ms     │ 100%        │ Danger 🔴
Cost AnalysisPlaintextBatch Size │ Cost/Request │ vs batch=8
───────────┼──────────────┼───────────
8          │ $0.000008    │ baseline
96         │ $0.006346    │ 793×
192        │ $0.025192    │ 3,149×
512        │ $0.050512    │ 6,314×
ML Model PerformancePlaintextModel          │ Precision │ Recall │ F1   │ AUC
───────────────┼───────────┼────────┼──────┼─────
Model A (raw)  │ 1.000     │ 1.000  │ 1.000│ 1.000
Model B (phys) │ 1.000     │ 1.000  │ 1.000│ 1.000
Best threshold │ 0.7       │ —      │ —    │ —
🛠️ Tech StackLayerTechnologyLanguagePython 3.12DataNumPy, PandasMLScikit-learn, MLflowPhysicsSciPy (M/M/1, Roofline)APIFastAPI, UvicornInfrastructureDocker, PostgreSQLMonitoringPrometheus, GrafanaVisualizationTableau, MatplotlibLLMGemini 1.5 Flash🚀 How to Run1. Clone the repoBashgit clone [https://github.com/gitdev77/system-performance-intelligence-platform](https://github.com/gitdev77/system-performance-intelligence-platform)
cd system-performance-intelligence-platform
2. Install dependenciesBashpip install -r requirements.txt
3. Run the FastAPI serverBashpython fastapiwrapper.py
# API live at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
4. Start monitoring stackBashpython day36_exporter.py &
docker-compose up -d
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)
5. Launch MLflow UIBashmlflow ui
# MLflow: http://localhost:5000
📁 Project StructurePlaintext├── telemetry_v2.csv          # 104-row dataset
├── day24_train.csv           # ML train split (83 rows)
├── day24_test.csv            # ML test split (21 rows)
├── fastapiwrapper.py         # FastAPI — 5 endpoints
├── day36_exporter.py         # Prometheus metrics exporter
├── day37_mlflow.py           # MLflow experiment tracking
├── docker-compose.yml        # Prometheus + Grafana stack
├── prometheus.yml            # Prometheus scrape config
├── dockerfile                # Docker image definition
├── requirements.txt          # Python dependencies
└── SQL.py                    # PostgreSQL analytics layer
📈 Daily ProgressDayTopicStatus1-7Telemetry foundation✅8-14Performance analytics✅15-21Bottleneck intelligence✅22-24Physics layer✅25-26ML model✅27-29MLOps monitoring✅30-31Cost modeling✅32FastAPI✅33Docker✅34PostgreSQL✅35Tableau dashboard✅36Prometheus + Grafana✅37MLflow tracking✅38Architecture diagram✅39README✅40Pytest suite🔄41-43LLM Layer (Gemini)🔄44GitHub finalization🔄🔗 AuthorDevansh — Building in public, one day at a time.GitHub: @gitdev77Built with Python, physics, and persistence. 44 days of daily commits.
