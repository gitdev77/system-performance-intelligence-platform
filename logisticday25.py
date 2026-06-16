import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (confusion_matrix, classification_report,
                                     precision_score, recall_score,
                                     f1_score, roc_auc_score)

# ─────────────────────────────────────────
# DAY 25 — LOGISTIC REGRESSION
# SLA Breach Prediction
#
# We train TWO models and compare them:
# Model A = raw features only
# Model B = raw + physics-informed (rho, AI)
#
# Goal: does adding physics features actually
# help the model predict SLA breaches better?
# ─────────────────────────────────────────

TRAIN_FILE = "/Users/devansh/Desktop/optimizeation/day24_train.csv"
TEST_FILE  = "/Users/devansh/Desktop/optimizeation/day24_test.csv"
OUTPUT_PNG = "/Users/devansh/Desktop/optimizeation/day25_ml_results.png"

# ── STEP 1: Load train and test sets ──
train = pd.read_csv(TRAIN_FILE)
test  = pd.read_csv(TEST_FILE)

TARGET = "sla_breach_new"

FEATURES_A = ["batch_size", "cpu_usage_clean",
              "throughput", "cv_latency"]

FEATURES_B = ["batch_size", "cpu_usage_clean",
              "throughput", "cv_latency",
              "rho", "AI"]

X_train_A = train[FEATURES_A]
X_train_B = train[FEATURES_B]
y_train   = train[TARGET]

X_test_A  = test[FEATURES_A]
X_test_B  = test[FEATURES_B]
y_test    = test[TARGET]

print(f"✅ Step 1 — Data loaded")
print(f"   Train: {len(train)} rows | Test: {len(test)} rows")
print(f"   Train breaches: {y_train.sum()} | "
      f"Test breaches: {y_test.sum()}\n")

# ── STEP 2: Scale features ──
# Fit scaler ONLY on train data.
# Then apply SAME scaler to test.
# (Important: never fit on test — that would be
#  like memorizing the exam answers beforehand)
scaler_A = StandardScaler()
scaler_B = StandardScaler()

X_train_A_sc = scaler_A.fit_transform(X_train_A)
X_test_A_sc  = scaler_A.transform(X_test_A)

X_train_B_sc = scaler_B.fit_transform(X_train_B)
X_test_B_sc  = scaler_B.transform(X_test_B)

print(f"✅ Step 2 — Features scaled (StandardScaler)\n")

# ── STEP 3: Train both models ──
# class_weight='balanced' → tells the model to
# pay more attention to rare breach examples
# max_iter=1000 → give gradient descent enough
# steps to find the best weights

model_A = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)
model_B = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model_A.fit(X_train_A_sc, y_train)
model_B.fit(X_train_B_sc, y_train)

print(f"✅ Step 3 — Both models trained\n")

# ── STEP 4: Predict on test set ──
y_pred_A  = model_A.predict(X_test_A_sc)
y_pred_B  = model_B.predict(X_test_B_sc)

# predict_proba gives actual probability scores
# (not just 0/1) — used for ROC-AUC
y_prob_A  = model_A.predict_proba(X_test_A_sc)[:, 1]
y_prob_B  = model_B.predict_proba(X_test_B_sc)[:, 1]

# ── STEP 5: Compute all metrics ──
def get_metrics(y_true, y_pred, y_prob, name):
    cm        = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    accuracy  = (y_true == y_pred).mean()
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    return {
        "name"     : name,
        "cm"       : cm,
        "tn"       : tn, "fp": fp,
        "fn"       : fn, "tp": tp,
        "accuracy" : accuracy,
        "precision": precision,
        "recall"   : recall,
        "f1"       : f1,
        "auc"      : auc,
    }

metrics_A = get_metrics(y_test, y_pred_A, y_prob_A,
                        "Model A (raw features)")
metrics_B = get_metrics(y_test, y_pred_B, y_prob_B,
                        "Model B (physics-informed)")

print(f"✅ Step 5 — Metrics computed\n")

# ── STEP 6: Feature weights (coefficients) ──
# In logistic regression, the weight (coefficient)
# of each feature after scaling tells you:
# "how much does this feature push the prediction
#  toward breach (positive) or no-breach (negative)?"
# Larger absolute value = stronger influence.
coef_A = pd.Series(
    model_A.coef_[0], index=FEATURES_A
).sort_values(key=abs, ascending=False)

coef_B = pd.Series(
    model_B.coef_[0], index=FEATURES_B
).sort_values(key=abs, ascending=False)

# ─────────────────────────────────────────
# STEP 7 — PLOT (4 panels)
# ─────────────────────────────────────────
colors = {
    "A"      : "#534AB7",
    "B"      : "#1D9E75",
    "tp"     : "#1D9E75",
    "tn"     : "#76C893",
    "fp"     : "#EF9F27",
    "fn"     : "#D85A30",
    "pos"    : "#1D9E75",
    "neg"    : "#D85A30",
    "neutral": "#999999",
}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "Day 25 — Logistic Regression: SLA Breach Prediction\n"
    "Model A (raw features) vs Model B (physics-informed)",
    fontsize=13, fontweight="bold", y=1.01
)

# ── PANEL 1 — Confusion matrices side by side ──
ax1 = axes[0][0]

# Draw 2 side-by-side 2x2 grids manually
def draw_cm(ax, metrics, x_offset, title, color):
    cells = [
        (0,0, metrics["tn"],  "TN\n(correct\nno-breach)", colors["tn"]),
        (0,1, metrics["fp"],  "FP\n(false\nalarm)",       colors["fp"]),
        (1,0, metrics["fn"],  "FN\n(missed\nbreach!)",    colors["fn"]),
        (1,1, metrics["tp"],  "TP\n(caught\nbreach)",     colors["tp"]),
    ]
    for row, col, val, label, c in cells:
        rect = plt.Rectangle(
            [x_offset + col * 0.45, 1 - row * 0.45],
            0.42, 0.42,
            facecolor=c, alpha=0.7,
            transform=ax.transData, clip_on=False
        )
        ax.add_patch(rect)
        ax.text(
            x_offset + col * 0.45 + 0.21,
            1 - row * 0.45 + 0.21,
            f"{val}\n{label}",
            ha="center", va="center",
            fontsize=8, fontweight="bold"
        )
    ax.text(
        x_offset + 0.45, 1.55, title,
        ha="center", fontsize=9,
        fontweight="bold", color=color
    )

ax1.set_xlim(0, 2.2)
ax1.set_ylim(0, 1.8)
ax1.axis("off")
draw_cm(ax1, metrics_A, 0.0,
        "Model A\n(raw features)", colors["A"])
draw_cm(ax1, metrics_B, 1.1,
        "Model B\n(physics-informed)", colors["B"])
ax1.set_title("Confusion Matrices", fontsize=11, fontweight="bold")

# ── PANEL 2 — Metrics comparison bar chart ──
ax2 = axes[0][1]
metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
vals_A = [metrics_A["accuracy"], metrics_A["precision"],
          metrics_A["recall"],   metrics_A["f1"],
          metrics_A["auc"]]
vals_B = [metrics_B["accuracy"], metrics_B["precision"],
          metrics_B["recall"],   metrics_B["f1"],
          metrics_B["auc"]]

x = np.arange(len(metric_names))
width = 0.35

bars_A = ax2.bar(x - width/2, vals_A, width,
                 color=colors["A"], alpha=0.8,
                 label="Model A (raw)")
bars_B = ax2.bar(x + width/2, vals_B, width,
                 color=colors["B"], alpha=0.8,
                 label="Model B (physics)")

for bar in bars_A:
    h = bar.get_height()
    if not np.isnan(h):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 h + 0.01, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=8)
for bar in bars_B:
    h = bar.get_height()
    if not np.isnan(h):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 h + 0.01, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=8)

ax2.set_xticks(x)
ax2.set_xticklabels(metric_names)
ax2.set_ylim(0, 1.15)
ax2.axhline(y=0.933, color=colors["fn"], linewidth=1.5,
            linestyle="--",
            label="Naive baseline (always predict 0) = 93.3%")
ax2.set_title("Metrics Comparison", fontsize=11, fontweight="bold")
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3, axis="y")

# ── PANEL 3 — Feature weights Model A ──
ax3 = axes[1][0]
bar_colors3 = [colors["pos"] if v > 0 else colors["neg"]
               for v in coef_A.values]
ax3.barh(coef_A.index, coef_A.values, color=bar_colors3, alpha=0.85)
ax3.axvline(x=0, color="black", linewidth=1)
ax3.set_title("Model A — Feature Weights\n"
              "(green = pushes toward breach, "
              "red = pushes away)",
              fontsize=10, fontweight="bold")
ax3.set_xlabel("Weight (coefficient after scaling)")
ax3.grid(True, alpha=0.3, axis="x")

# ── PANEL 4 — Feature weights Model B ──
ax4 = axes[1][1]
bar_colors4 = [colors["pos"] if v > 0 else colors["neg"]
               for v in coef_B.values]
ax4.barh(coef_B.index, coef_B.values, color=bar_colors4, alpha=0.85)
ax4.axvline(x=0, color="black", linewidth=1)
ax4.set_title("Model B — Feature Weights\n"
              "(green = pushes toward breach, "
              "red = pushes away)",
              fontsize=10, fontweight="bold")
ax4.set_xlabel("Weight (coefficient after scaling)")
ax4.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"✅ Chart saved → {OUTPUT_PNG}\n")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("=" * 65)
print("  DAY 25 FINDINGS — LOGISTIC REGRESSION RESULTS")
print("=" * 65)

for m in [metrics_A, metrics_B]:
    print(f"\n── {m['name']} ──\n")
    print(f"  Accuracy  : {m['accuracy']:.3f}  "
          f"(naive baseline = 0.933)")
    print(f"  Precision : {m['precision']:.3f}  "
          f"(of predicted breaches, how many were real?)")
    print(f"  Recall    : {m['recall']:.3f}  "
          f"(of real breaches, how many did we catch?)")
    print(f"  F1 Score  : {m['f1']:.3f}  "
          f"(balance of precision + recall)")
    print(f"  AUC       : {m['auc']:.3f}  "
          f"(1.0 = perfect, 0.5 = random guess)")
    print(f"\n  Confusion:")
    print(f"  TN={m['tn']} (correct no-breach)  "
          f"FP={m['fp']} (false alarms)")
    print(f"  FN={m['fn']} (missed breaches!)  "
          f"TP={m['tp']} (caught breaches)")

print(f"\n── Feature Weights — Model A (raw only) ──\n")
for feat, val in coef_A.items():
    direction = "→ breach" if val > 0 else "→ no breach"
    print(f"  {feat:<20} {val:+.3f}  {direction}")

print(f"\n── Feature Weights — Model B (physics-informed) ──\n")
for feat, val in coef_B.items():
    direction = "→ breach" if val > 0 else "→ no breach"
    print(f"  {feat:<20} {val:+.3f}  {direction}")

print(f"\n── Physics Features Verdict ──\n")
recall_diff = metrics_B["recall"] - metrics_A["recall"]
f1_diff     = metrics_B["f1"]     - metrics_A["f1"]
auc_diff    = metrics_B["auc"]    - metrics_A["auc"]

print(f"  Recall improvement  : {recall_diff:+.3f}")
print(f"  F1 improvement      : {f1_diff:+.3f}")
print(f"  AUC improvement     : {auc_diff:+.3f}")

if recall_diff > 0 or auc_diff > 0.02:
    print(f"\n  ✅ Physics-informed features (ρ, AI) IMPROVED the model.")
    print(f"     Adding queuing theory + ECE hardware features gives")
    print(f"     the model better signal for catching SLA breaches.")
elif recall_diff == 0 and abs(f1_diff) < 0.05:
    print(f"\n  ➡️  Physics features made NO significant difference.")
    print(f"     Raw features (especially cv_latency + cpu) already")
    print(f"     capture most of the signal. This is an honest finding")
    print(f"     — the physics layer's value is in EXPLANATION (Days")
    print(f"     21-23), not necessarily in raw ML predictive power.")
else:
    print(f"\n  ⚠️  Mixed result — check confusion matrices carefully.")

print(f"\n── Important Caveat ──\n")
print(f"  Test set has only {len(y_test)} rows with "
      f"{y_test.sum()} breach(es).")
print(f"  Results are directionally correct but not")
print(f"  statistically robust at this dataset size.")
print(f"  Real production systems need 100s of breach")
print(f"  examples for reliable ML training.")

print(f"\n{'='*65}")
print(f"  Day 25 complete. Output → day25_ml_results.png")
print(f"{'='*65}\n")