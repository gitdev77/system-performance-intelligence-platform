# day32_api.py
# Run with: uvicorn day32_api:app --reload --port 8000
# Then open: http://localhost:8000/docs

import pandas as pd
import numpy as np
import json
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_FILE  = "/Users/devansh/Desktop/optimizeation/telemetry_v2.csv"
TRAIN_FILE = "/Users/devansh/Desktop/optimizeation/day24_train.csv"
COST_FILE  = "/Users/devansh/Desktop/optimizeation/day30_cost_report.json"
REC_FILE   = "/Users/devansh/Desktop/optimizeation/day31_recommendation.json"

SLA_THRESHOLD_MS = 20.0
ALERT_THRESHOLD  = 0.7
MATRIX_DIM       = 512
BYTES_PER_FLOAT  = 4
CPU_COST_PER_SEC = 0.05 / 3600
SLA_PENALTY      = 0.50

# ─────────────────────────────────────────
# GLOBAL STATE
# (loaded once at startup, reused forever)
# ─────────────────────────────────────────
state = {
    "model"         : None,
    "scaler"        : None,
    "agg"           : None,   # aggregated telemetry
    "cost_report"   : None,
    "recommendation": None,
    "startup_time"  : None,
}

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# (same logic as Day 20 bottleneck engine
#  and Day 30 cost model — reused here)
# ─────────────────────────────────────────

def prepare_data():
    """Load telemetry, clean, aggregate."""
    df = pd.read_csv(DATA_FILE)

    df["cpu_usage_clean"] = df["cpu_usage"].replace(0.0, np.nan)
    df["cpu_usage_clean"] = (
        df["cpu_usage_clean"]
        .rolling(window=5, min_periods=1, center=True)
        .median().ffill().bfill()
    )
    df["avg_latency_ms"] = df["avg_latency"] * 1000
    df["sla_breach_new"] = (
        df["avg_latency_ms"] > SLA_THRESHOLD_MS
    ).astype(int)

    throughput_peak = (
        df.groupby("batch_size")["throughput"]
        .mean().max()
    )
    df["rho"] = (
        1 - df["throughput"] / throughput_peak
    ).clip(lower=0)

    b           = df["batch_size"]
    flops       = 2 * b * MATRIX_DIM * MATRIX_DIM
    bytes_moved = (
        (b * MATRIX_DIM) +
        (MATRIX_DIM * MATRIX_DIM) +
        (b * MATRIX_DIM)
    ) * BYTES_PER_FLOAT
    df["AI"] = flops / bytes_moved

    agg = (
        df.groupby("batch_size")
        .agg(
            avg_latency_ms  = ("avg_latency_ms",  "mean"),
            throughput      = ("throughput",       "mean"),
            cpu_usage       = ("cpu_usage_clean",  "mean"),
            sla_breach_rate = ("sla_breach_new",   "mean"),
            cv_latency      = ("cv_latency",       "mean"),
            rho             = ("rho",              "mean"),
            AI              = ("AI",               "mean"),
        )
        .reset_index()
        .sort_values("batch_size")
        .reset_index(drop=True)
    )

    agg["sla_breach_pct"] = agg["sla_breach_rate"] * 100

    # Cost metrics
    agg["latency_sec"]      = agg["avg_latency_ms"] / 1000
    agg["cost_per_request"] = (
        agg["latency_sec"] * CPU_COST_PER_SEC /
        agg.index.map(lambda i: agg.loc[i, "batch_size"])
    )
    agg["cost_per_request"] = (
        (agg["latency_sec"] * CPU_COST_PER_SEC) /
        agg["batch_size"]
    )
    agg["penalty_per_req"]  = (
        agg["sla_breach_rate"] * SLA_PENALTY /
        agg["batch_size"]
    )
    agg["total_cost"]       = (
        agg["cost_per_request"] + agg["penalty_per_req"]
    )

    return agg, df["throughput"].max()


def detect_bottleneck_logic(row):
    """
    Simplified bottleneck detection — same rules
    as Day 20 engine, extracted as a pure function
    so the API can call it per-request.
    """
    tput_collapse = (
        1 - row["throughput"] /
        state["agg"]["throughput"].max()
    ) * 100
    lat_growth    = (
        row["avg_latency_ms"] /
        state["agg"]["avg_latency_ms"].min()
    )
    cpu_stressed  = row["cpu_usage"] > 75.0
    high_variance = row["cv_latency"] > 0.35
    sla_bad       = row["sla_breach_pct"] > 50.0

    if tput_collapse > 35 and lat_growth > 1.8:
        btype    = "throughput_collapse"
        severity = "critical" if sla_bad else "high"
    elif cpu_stressed and lat_growth > 1.8:
        btype    = "cpu_saturation"
        severity = "high"
    elif high_variance and not cpu_stressed:
        btype    = "scheduling_noise"
        severity = "medium"
    elif tput_collapse > 20:
        btype    = "degrading"
        severity = "medium"
    else:
        btype    = "healthy"
        severity = "healthy"

    return btype, severity


def train_model():
    """Train logistic regression on Day 24 data."""
    train    = pd.read_csv(TRAIN_FILE)
    FEATURES = [
        "batch_size", "cpu_usage_clean",
        "throughput", "cv_latency", "rho", "AI"
    ]
    TARGET   = "sla_breach_new"

    scaler   = StandardScaler()
    X_tr     = scaler.fit_transform(train[FEATURES])
    y_tr     = train[TARGET]

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    model.fit(X_tr, y_tr)

    return model, scaler

# ─────────────────────────────────────────
# LIFESPAN — runs once at startup
# (loads model + data so requests are fast)
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Everything inside the 'with' block runs
    at startup. The model and data are loaded
    ONCE here, then reused for every request.
    This is critical for performance — you
    never want to retrain a model mid-request.
    """
    print("🚀 Starting System Performance Intelligence API...")

    state["agg"], _      = prepare_data()
    state["model"], state["scaler"] = train_model()
    state["startup_time"] = datetime.now().isoformat()

    # Load pre-computed outputs from Day 30/31
    try:
        with open(COST_FILE) as f:
            state["cost_report"] = json.load(f)
    except FileNotFoundError:
        state["cost_report"] = None

    try:
        with open(REC_FILE) as f:
            state["recommendation"] = json.load(f)
    except FileNotFoundError:
        state["recommendation"] = None

    print("✅ Model trained and data loaded.")
    print("✅ API ready at http://localhost:8000")
    print("📖 Docs at http://localhost:8000/docs\n")

    yield   # API runs here (between startup and shutdown)

    print("👋 Shutting down API.")

# ─────────────────────────────────────────
# CREATE THE FASTAPI APP
# ─────────────────────────────────────────
app = FastAPI(
    title       = "System Performance Intelligence API",
    description = (
        "Physics-informed ML system for SLA breach "
        "prediction, bottleneck diagnosis, and "
        "cost-optimal batch size recommendation.\n\n"
        "Built on M/M/1 queuing theory (r=0.974) + "
        "Roofline ECE model + Logistic Regression."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ─────────────────────────────────────────
# PYDANTIC SCHEMAS
# (define exactly what shape each request
#  body must have — FastAPI validates this
#  automatically before your code runs)
# ─────────────────────────────────────────

class TelemetryInput(BaseModel):
    """Input schema for /predict endpoint."""
    batch_size      : int   = Field(
        ..., ge=1, le=1024,
        description="Batch size (1-1024)"
    )
    cpu_usage_clean : float = Field(
        ..., ge=0, le=100,
        description="CPU usage % (0-100)"
    )
    throughput      : float = Field(
        ..., ge=0,
        description="Throughput (requests/sec)"
    )
    cv_latency      : float = Field(
        ..., ge=0,
        description="Coefficient of variation of latency"
    )

    model_config = {
    "json_schema_extra": {
        "example": {
            "batch_size"      : 192,
            "cpu_usage_clean" : 68.3,
            "throughput"      : 31.95,
            "cv_latency"      : 0.342
        }
    }
}


class DiagnoseInput(BaseModel):
    """Input schema for /diagnose endpoint."""
    batch_size: int = Field(
        ..., ge=1, le=1024,
        description="Batch size to diagnose"
    )

    model_config = {
    "json_schema_extra": {"example": {"batch_size": 192}}
}
# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

# ── 1. HEALTH CHECK ──
@app.get(
    "/health",
    summary="Health check",
    tags=["System"]
)
def health_check():
    """
    Returns service status and basic metadata.
    Use this to verify the API is running.
    """
    return {
        "status"      : "healthy",
        "version"     : "1.0.0",
        "startup_time": state["startup_time"],
        "model_loaded": state["model"] is not None,
        "data_loaded" : state["agg"] is not None,
        "n_batch_sizes": (
            len(state["agg"])
            if state["agg"] is not None else 0
        ),
        "timestamp"   : datetime.now().isoformat(),
        "project"     : (
            "System Performance Intelligence Platform"
        ),
    }


# ── 2. PREDICT SLA BREACH ──
@app.post(
    "/predict",
    summary="Predict SLA breach probability",
    tags=["ML Prediction"]
)
def predict_breach(payload: TelemetryInput):
    """
    Takes live telemetry and returns the
    probability that this run will breach SLA.

    Physics-informed features (rho, AI) are
    computed automatically from the inputs —
    you don't need to supply them.

    Returns breach probability + alert decision.
    """
    if state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    # Derive physics features automatically
    throughput_peak = state["agg"]["throughput"].max()
    rho = max(
        0,
        1 - payload.throughput / throughput_peak
    )

    b           = payload.batch_size
    flops       = 2 * b * MATRIX_DIM * MATRIX_DIM
    bytes_moved = (
        (b * MATRIX_DIM) +
        (MATRIX_DIM * MATRIX_DIM) +
        (b * MATRIX_DIM)
    ) * BYTES_PER_FLOAT
    AI = flops / bytes_moved

    # Build feature vector
    features = pd.DataFrame([{
        "batch_size"      : payload.batch_size,
        "cpu_usage_clean" : payload.cpu_usage_clean,
        "throughput"      : payload.throughput,
        "cv_latency"      : payload.cv_latency,
        "rho"             : rho,
        "AI"              : AI,
    }])

    features_scaled = state["scaler"].transform(features)
    breach_prob     = float(
        state["model"].predict_proba(
            features_scaled
        )[0][1]
    )
    alert_fired = breach_prob >= ALERT_THRESHOLD

    # Log level
    if alert_fired:
        level = "ALERT"
    elif breach_prob >= 0.4:
        level = "WARNING"
    else:
        level = "INFO"

    return {
        "input": {
            "batch_size"     : payload.batch_size,
            "cpu_usage"      : payload.cpu_usage_clean,
            "throughput"     : payload.throughput,
            "cv_latency"     : payload.cv_latency,
        },
        "physics_features": {
            "rho": round(rho, 4),
            "AI" : round(AI, 2),
        },
        "prediction": {
            "breach_probability": round(breach_prob, 4),
            "alert_fired"       : alert_fired,
            "threshold_used"    : ALERT_THRESHOLD,
            "log_level"         : level,
        },
        "interpretation": (
            f"{'⚠️ ALERT: High breach risk' if alert_fired else '✅ Normal: Low breach risk'} "
            f"(p={breach_prob:.3f})"
        ),
        "timestamp": datetime.now().isoformat(),
    }


# ── 3. DIAGNOSE BOTTLENECK ──
@app.post(
    "/diagnose",
    summary="Diagnose bottleneck at a batch size",
    tags=["Bottleneck Intelligence"]
)
def diagnose(payload: DiagnoseInput):
    """
    Returns the bottleneck type and severity
    for a given batch size, using the Day 20
    rule-based detection engine.

    If the batch size isn't in the dataset,
    returns the nearest known batch size result.
    """
    agg = state["agg"]

    # Find nearest batch size in dataset
    available = agg["batch_size"].values
    nearest   = int(
        available[
            np.argmin(np.abs(
                available - payload.batch_size
            ))
        ]
    )
    row = agg[
        agg["batch_size"] == nearest
    ].iloc[0]

    btype, severity = detect_bottleneck_logic(row)

    return {
        "requested_batch_size": payload.batch_size,
        "nearest_in_dataset"  : nearest,
        "metrics": {
            "avg_latency_ms" : round(
                float(row["avg_latency_ms"]), 2),
            "throughput"     : round(
                float(row["throughput"]), 2),
            "cpu_usage"      : round(
                float(row["cpu_usage"]), 1),
            "cv_latency"     : round(
                float(row["cv_latency"]), 3),
            "sla_breach_pct" : round(
                float(row["sla_breach_pct"]), 1),
            "rho"            : round(
                float(row["rho"]), 3),
        },
        "diagnosis": {
            "bottleneck_type": btype,
            "severity"       : severity,
            "recommendation" : (
                "Reduce batch size or scale horizontally"
                if btype == "throughput_collapse" else
                "Add compute resources"
                if btype == "cpu_saturation" else
                "Avoid for latency-sensitive workloads"
                if btype == "scheduling_noise" else
                "Monitor closely"
                if btype == "degrading" else
                "System operating normally"
            ),
        },
        "timestamp": datetime.now().isoformat(),
    }


# ── 4. RECOMMEND OPTIMAL BATCH ──
@app.get(
    "/recommend",
    summary="Get optimal batch size recommendation",
    tags=["Optimization"]
)
def recommend():
    """
    Returns the Day 31 optimization engine's
    recommendation — which batch size to run
    in production, why, and the business impact.
    """
    if state["recommendation"] is None:
        # Fallback if JSON file not found
        return {
            "recommended_batch": 8,
            "note": (
                "day31_recommendation.json not found."
                " Run day31.py first for full analysis."
            )
        }

    rec = state["recommendation"]
    return {
        "recommendation"       : rec[
            "final_recommendation"
        ],
        "pareto_frontier"      : rec[
            "pareto_frontier_batches"
        ],
        "dominated_avoid"      : rec[
            "dominated_batches"
        ],
        "scenario_breakdown"   : rec[
            "scenario_results"
        ],
        "generated_at"         : rec.get(
            "generated_at", "N/A"
        ),
    }


# ── 5. COST BREAKDOWN ──
@app.get(
    "/cost/{batch_size}",
    summary="Get cost breakdown for a batch size",
    tags=["Cost Modeling"]
)
def cost_breakdown(batch_size: int):
    """
    Returns the full cost breakdown for a
    specific batch size — compute cost, SLA
    penalty cost, and total cost per request.

    If batch size not in dataset, returns
    the nearest available.
    """
    agg = state["agg"]

    available = agg["batch_size"].values
    nearest   = int(
        available[
            np.argmin(np.abs(
                available - batch_size
            ))
        ]
    )
    row = agg[
        agg["batch_size"] == nearest
    ].iloc[0]

    compute_cost = float(row["cost_per_request"])
    penalty_cost = float(row["penalty_per_req"])
    total_cost   = float(row["total_cost"])

    # Annual projection
    daily_reqs   = 1_000_000
    annual_cost  = total_cost * daily_reqs * 365

    return {
        "requested_batch_size": batch_size,
        "nearest_in_dataset"  : nearest,
        "cost_breakdown": {
            "compute_cost_usd"    : round(
                compute_cost, 10),
            "sla_penalty_usd"     : round(
                penalty_cost, 10),
            "total_cost_usd"      : round(
                total_cost, 10),
            "compute_cost_micro"  : round(
                compute_cost * 1e6, 4),
            "total_cost_micro"    : round(
                total_cost * 1e6, 4),
        },
        "projections": {
            "daily_cost_1m_reqs"  : round(
                total_cost * daily_reqs, 4),
            "annual_cost_1m_reqs" : round(
                annual_cost, 2),
        },
        "performance": {
            "avg_latency_ms"  : round(
                float(row["avg_latency_ms"]), 2),
            "sla_breach_pct"  : round(
                float(row["sla_breach_pct"]), 1),
            "throughput"      : round(
                float(row["throughput"]), 2),
        },
        "verdict": (
            "⚠️ AVOID — SLA breaches detected"
            if row["sla_breach_pct"] > 0
            else "✅ SAFE — No SLA breaches"
        ),
        "timestamp": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────
# RUN DIRECTLY (alternative to uvicorn CLI)
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "day32_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )