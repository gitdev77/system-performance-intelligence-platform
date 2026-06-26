
# System Performance Intelligence & Cost-Aware Optimization Platform

> 🚀 A production-grade ML system that predicts SLA breaches, detects performance cliffs, and optimizes batch size selection for ML inference workloads — saving **$356,441/year** at 1M daily requests.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![Docker](https://img.shields.io/badge/Docker-containerized-blue) ![MLflow](https://img.shields.io/badge/MLflow-tracked-orange) ![Grafana](https://img.shields.io/badge/Grafana-monitored-orange)

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
| Annual saving from optimal batch selection | **$356,441** |
| Recommended operating point | batch = 8 or 16 |

---

## 🏗️ System Architecture

```
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
```

---

## 🎯 Problem Statement

ML inference systems face a critical challenge: **batch size selection directly controls the cost-performance tradeoff**, yet most teams choose batch sizes arbitrarily.

This project answers: *At what batch size does the system cliff, why does it happen, and what is the optimal operating point?*

- **Workload:** Matrix multiply A(batch×512) × B(512×512), float32, Apple Silicon Mac
- **Dataset:** 104 telemetry runs across 13 batch sizes [8→512], 8 runs each

---

## 🔬 Approach

### Week 1-2: Telemetry Foundation
- Built psutil-based system monitor and workload simulator
- Generated 104-row dataset with 13 batch sizes
- Key discovery: throughput is primary latency driver (r=-0.982), CPU is NOT the bottleneck (r=-0.137)

### Week 3: Bottleneck Intelligence
- Rule-based bottleneck classifier + KMeans regime detection (K=3)
- Anomaly detection via Z-score + IQR
- Performance cliff confirmed at batch=192

### Physics Layer (Days 22-24)
- **M/M/1 Queuing Theory:** theoretical delay W(b) correlates r=0.974 with actual latency
- **Phase Transition Analysis:** critical point b_c=32 detected
- **ECE Roofline Model:** memory bandwidth ruled out as cliff cause

### ML Layer (Days 25-29)
- Physics-informed features (rho from M/M/1, AI from Roofline)
- Logistic Regression: perfect precision=1.0, recall=1.0, AUC=1.0
- MLflow tracked 9 experiments across 3 model variants and 3 thresholds
- Optimal threshold=0.7 gives zero false negatives across all sessions

### Cost Modeling (Days 30-31)
- Cost per request = compute cost + SLA penalty
- Pareto frontier analysis across 3 business scenarios
- All scenarios agree: batch=8 recommended

---

## 📊 Results

### Performance Cliff

| Batch Size | Avg Latency | P95 Latency | Breach Rate | Regime |
|-----------|-------------|-------------|-------------|--------|
| 8 | 7.15ms | 9.61ms | 0% | Safe ✅ |
| 96 | 10.16ms | 15.50ms | 12.5% | Safe ⚠️ |
| 128 | 11.87ms | 21.36ms | 25% | Warning ⚠️ |
| 192 | 18.50ms | 29.38ms | 50% | Danger 🔴 |
| 512 | 20.85ms | 28.61ms | 100% | Danger 🔴 |

### Cost Analysis

| Batch Size | Cost/Request | vs batch=8 |
|-----------|-------------|-----------|
| 8 | $0.000008 | baseline |
| 96 | $0.006346 | 793× |
| 192 | $0.025192 | 3,149× |
| 512 | $0.050512 | **6,314×** |

### ML Model Performance

| Model | Precision | Recall | F1 | AUC |
|-------|-----------|--------|-----|-----|
| Model A (raw features) | 1.000 | 1.000 | 1.000 | 1.000 |
| Model B (physics-informed) | 1.000 | 1.000 | 1.000 | 1.000 |
| Best threshold | 0.7 | — | — | — |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Data | NumPy, Pandas |
| ML | Scikit-learn, MLflow |
| Physics | SciPy (M/M/1, Roofline) |
| API | FastAPI, Uvicorn |
| Infrastructure | Docker, PostgreSQL |
| Monitoring | Prometheus, Grafana |
| Visualization | Tableau, Matplotlib |
| LLM | Gemini 1.5 Flash |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/gitdev77/system-performance-intelligence-platform
cd system-performance-intelligence-platform
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the FastAPI server
```bash
python fastapiwrapper.py
# API live at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

### 4. Start monitoring stack
```bash
python day36_exporter.py &
docker-compose up -d
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (login: admin / admin)
```

### 5. Launch MLflow UI
```bash
mlflow ui
# MLflow: http://localhost:5000
```

---

## 📁 Project Structure

```
├── telemetry_v2.csv          # 104-row dataset (13 batch sizes, 8 runs each)
├── day24_train.csv           # ML train split (83 rows, stratified)
├── day24_test.csv            # ML test split (21 rows, stratified)
├── fastapiwrapper.py         # FastAPI — 5 endpoints
├── day36_exporter.py         # Prometheus metrics exporter (port 8001)
├── day37_mlflow.py           # MLflow experiment tracking (9 runs)
├── docker-compose.yml        # Prometheus + Grafana stack
├── prometheus.yml            # Prometheus scrape config
├── dockerfile                # Docker image definition
├── requirements.txt          # Python dependencies
└── SQL.py                    # PostgreSQL analytics layer
```

---

## 📈 Daily Progress

| Day | Topic | Status |
|-----|-------|--------|
| 1–7 | Telemetry foundation | ✅ |
| 8–14 | Performance analytics | ✅ |
| 15–21 | Bottleneck intelligence | ✅ |
| 22–24 | Physics layer | ✅ |
| 25–26 | ML model | ✅ |
| 27–29 | MLOps monitoring | ✅ |
| 30–31 | Cost modeling | ✅ |
| 32 | FastAPI | ✅ |
| 33 | Docker | ✅ |
| 34 | PostgreSQL | ✅ |
| 35 | Tableau dashboard | ✅ |
| 36 | Prometheus + Grafana | ✅ |
| 37 | MLflow tracking | ✅ |
| 38 | Architecture diagram | ✅ |
| 39 | README | ✅ |
| 40 | Pytest suite | 🔄 |
| 41–43 | LLM Layer (Gemini) | 🔄 |
| 44 | GitHub finalization | 🔄 |

---

## 🔗 Author

**Devansh** — Building in public, one day at a time.
GitHub: [@gitdev77](https://github.com/gitdev77)

---

*Built with Python, physics, and persistence. 44 days of daily commits.*