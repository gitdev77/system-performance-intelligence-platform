# day37_mlflow.py
# Day 37: MLflow Experiment Tracking
# Re-runs your Days 25-29 ML experiments and logs everything to MLflow

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, roc_auc_score,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load your existing train/test splits from Day 24 ───────────────────
print("Loading Day 24 train/test data...")
train = pd.read_csv("day24_train.csv")
test  = pd.read_csv("day24_test.csv")
print(f"Train: {len(train)} rows | Test: {len(test)} rows")
print(f"Columns: {list(train.columns)}")

# ── 2. Define features ────────────────────────────────────────────────────
# Model A: raw features only (no physics)
FEATURES_A = ["batch_size", "throughput", "cpu_usage_clean", "cv_latency"]

FEATURES_B = ["batch_size", "throughput", "cpu_usage_clean",
              "cv_latency", "rho", "AI"]

TARGET = "sla_breach_new"

# ── 3. Set up MLflow experiment ───────────────────────────────────────────
mlflow.set_experiment("SLA_Breach_Prediction")
print("\n✅ MLflow experiment: 'SLA_Breach_Prediction'")

# ── 4. Helper function — train and log one model ──────────────────────────
def run_experiment(run_name, features, threshold, C=1.0):
    """Train logistic regression and log everything to MLflow."""

    # Filter to available features
    available = [f for f in features if f in train.columns]
    print(f"\n── {run_name} ──────────────────────────────────")
    print(f"   Features: {available}")
    print(f"   Threshold: {threshold} | C: {C}")

    X_train = train[available]
    y_train = train[TARGET]
    X_test  = test[available]
    y_test  = test[TARGET]

    # Scale features
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(X_train_sc, y_train)

    # Predict with custom threshold
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() \
                     if len(np.unique(y_pred)) > 1 \
                     else (sum(y_test==0), 0, 0, sum(y_test==1))

    # ── LOG TO MLFLOW ──────────────────────────────────────────────────
    with mlflow.start_run(run_name=run_name):

        # Log parameters
        mlflow.log_param("features",       str(available))
        mlflow.log_param("n_features",     len(available))
        mlflow.log_param("threshold",      threshold)
        mlflow.log_param("C",              C)
        mlflow.log_param("model_type",     "LogisticRegression")
        mlflow.log_param("physics_informed",
                         "rho" in available)

        # Log metrics
        mlflow.log_metric("precision",     precision)
        mlflow.log_metric("recall",        recall)
        mlflow.log_metric("f1_score",      f1)
        mlflow.log_metric("auc",           auc)
        mlflow.log_metric("true_positives",  float(tp))
        mlflow.log_metric("false_positives", float(fp))
        mlflow.log_metric("false_negatives", float(fn))
        mlflow.log_metric("true_negatives",  float(tn))

        # Log feature importances as metrics
        for fname, coef in zip(available, model.coef_[0]):
            mlflow.log_metric(f"coef_{fname}", round(float(coef), 4))

        # Log the model itself
        mlflow.sklearn.log_model(model, "model")

    # Print results
    print(f"   precision={precision:.3f} | recall={recall:.3f} | "
          f"f1={f1:.3f} | AUC={auc:.3f}")
    print(f"   TP={tp} FP={fp} FN={fn} TN={tn}")

    return precision, recall, f1, auc

# ── 5. Run all experiments ────────────────────────────────────────────────
print("\n" + "="*60)
print("RUNNING ALL EXPERIMENTS")
print("="*60)

results = []

# Model A — raw features, 3 thresholds
for thresh in [0.3, 0.5, 0.7]:
    p, r, f, a = run_experiment(
        run_name=f"ModelA_raw_thresh{thresh}",
        features=FEATURES_A,
        threshold=thresh
    )
    results.append(("Model A", thresh, p, r, f, a))

# Model B — physics features, 3 thresholds
for thresh in [0.3, 0.5, 0.7]:
    p, r, f, a = run_experiment(
        run_name=f"ModelB_physics_thresh{thresh}",
        features=FEATURES_B,
        threshold=thresh
    )
    results.append(("Model B", thresh, p, r, f, a))

# Model C — physics features, tuned C parameter
for C_val in [0.1, 1.0, 10.0]:
    p, r, f, a = run_experiment(
        run_name=f"ModelC_physics_C{C_val}",
        features=FEATURES_B,
        threshold=0.7,
        C=C_val
    )
    results.append((f"Model C (C={C_val})", 0.7, p, r, f, a))

# ── 6. Summary table ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
print(f"{'Run':<30} {'Thresh':>6} {'Prec':>6} {'Rec':>6} "
      f"{'F1':>6} {'AUC':>6}")
print("-"*60)
for model, thresh, p, r, f, a in results:
    print(f"{model:<30} {thresh:>6.1f} {p:>6.3f} {r:>6.3f} "
          f"{f:>6.3f} {a:>6.3f}")

print("\n✅ All runs logged to MLflow!")
print("   Start the UI with: mlflow ui")
print("   Then open: http://localhost:5000")
