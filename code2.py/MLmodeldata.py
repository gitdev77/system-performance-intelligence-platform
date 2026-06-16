import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────
# DAY 24 — ML DATASET PREP
# Physics-Informed Feature Engineering
#
# GOAL: Build one clean table (X = features,
# y = target) ready for Day 25's logistic
# regression — using physics (rho) and
# ECE (AI) features alongside raw telemetry.
# ─────────────────────────────────────────

DATA_FILE   = "/Users/devansh/Desktop/optimizeation/telemetry_v2.csv"
TRAIN_OUT   = "/Users/devansh/Desktop/optimizeation/day24_train.csv"
TEST_OUT    = "/Users/devansh/Desktop/optimizeation/day24_test.csv"

SLA_THRESHOLD_MS = 20.0   # same threshold used since Day 12
MATRIX_DIM       = 512    # same as Day 23
BYTES_PER_FLOAT  = 4

# ── STEP 1: Load raw (per-run) telemetry ──
df = pd.read_csv(DATA_FILE)
print(f"✅ Step 1 — Loaded {len(df)} individual runs "
      f"(13 batch sizes × 8 runs)\n")

# ── STEP 2: Clean CPU (same rolling median as Day 19-23) ──
df["cpu_usage_clean"] = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage_clean"] = (
    df["cpu_usage_clean"]
    .rolling(window=5, min_periods=1, center=True)
    .median().ffill().bfill()
)

# ── STEP 3: Convert latency to ms ──
df["avg_latency_ms"] = df["avg_latency"] * 1000

# ── STEP 4: Recompute SLA breach target (y) ──
# OLD column "sla_breach" is useless (0% everywhere —
# baked in with a 150ms threshold that nothing crosses).
# We recompute using OUR 20ms threshold, per-run.
df["sla_breach_new"] = (df["avg_latency_ms"] > SLA_THRESHOLD_MS).astype(int)

n_breach    = df["sla_breach_new"].sum()
n_total     = len(df)
breach_pct  = (n_breach / n_total) * 100

print(f"✅ Step 4 — Target recomputed (20ms threshold)")
print(f"   Old 'sla_breach' column: "
      f"{df['sla_breach'].sum()}/{n_total} positive (USELESS)")
print(f"   New 'sla_breach_new'   : "
      f"{n_breach}/{n_total} positive ({breach_pct:.1f}%)\n")

# ── STEP 5: Physics feature — rho(b) per row ──
# Same formula as Day 21: rho = 1 - (throughput / peak)
# peak = best-ever average throughput (across batch sizes)
throughput_peak = df.groupby("batch_size")["throughput"].mean().max()
df["rho"] = (1 - (df["throughput"] / throughput_peak)).clip(lower=0)

print(f"✅ Step 5 — Physics feature ρ(b) added "
      f"(peak throughput reference = {throughput_peak:.2f})")

# ── STEP 6: ECE feature — Arithmetic Intensity AI(b) ──
# Same formula as Day 23. Depends only on batch_size,
# so identical for all 8 runs within a batch size —
# but it still encodes useful structural information.
b = df["batch_size"]
flops       = 2 * b * MATRIX_DIM * MATRIX_DIM
bytes_moved = ((b * MATRIX_DIM) + (MATRIX_DIM * MATRIX_DIM) + (b * MATRIX_DIM)) \
              * BYTES_PER_FLOAT
df["AI"] = flops / bytes_moved

print(f"✅ Step 6 — ECE feature AI(b) added\n")

# ── STEP 7: Assemble X (features) and y (target) ──
feature_cols = [
    "batch_size",
    "cpu_usage_clean",
    "throughput",
    "cv_latency",
    "rho",     # physics-informed
    "AI",      # ECE-informed
]
target_col = "sla_breach_new"

ml_df = df[feature_cols + [target_col]].copy()

print(f"✅ Step 7 — Feature matrix assembled")
print(f"   X columns (features): {feature_cols}")
print(f"   y column  (target)  : '{target_col}'")
print(f"   Dropped (would leak the answer): "
      f"avg_latency, avg_latency_ms, p95_latency, p99_latency, "
      f"old sla_breach\n")

# ── STEP 8: Class balance check ──
print(f"✅ Step 8 — Class balance")
print(f"   No breach (0): {(ml_df[target_col]==0).sum()} rows "
      f"({(ml_df[target_col]==0).mean()*100:.1f}%)")
print(f"   Breach    (1): {(ml_df[target_col]==1).sum()} rows "
      f"({(ml_df[target_col]==1).mean()*100:.1f}%)")

if breach_pct < 15 or breach_pct > 85:
    print(f"   ⚠️  Imbalanced! A model that always predicts the "
          f"majority class")
    print(f"      would score "
          f"{max(breach_pct, 100-breach_pct):.1f}% accuracy "
          f"WITHOUT learning anything.")
    print(f"      → On Day 25 we must check PRECISION/RECALL, "
          f"not just accuracy.")
print()

# ── STEP 9: Train/test split (80/20, stratified) ──
X = ml_df[feature_cols]
y = ml_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,     # makes the split reproducible every run
    stratify=y           # keep breach % similar in train AND test
)

print(f"✅ Step 9 — Train/test split (80/20, stratified)")
print(f"   Train: {len(X_train)} rows "
      f"({y_train.mean()*100:.1f}% breach)")
print(f"   Test : {len(X_test)} rows "
      f"({y_test.mean()*100:.1f}% breach)\n")

# ── STEP 10: Save for Day 25 ──
train_df = X_train.copy()
train_df[target_col] = y_train
test_df = X_test.copy()
test_df[target_col] = y_test

train_df.to_csv(TRAIN_OUT, index=False)
test_df.to_csv(TEST_OUT, index=False)

print(f"✅ Step 10 — Saved")
print(f"   Train → {TRAIN_OUT}")
print(f"   Test  → {TEST_OUT}\n")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("=" * 65)
print("  DAY 24 FINDINGS — ML DATASET READY")
print("=" * 65)

print(f"\n── Feature Summary (full dataset, n={len(ml_df)}) ──\n")
print(ml_df[feature_cols].describe().round(3).to_string())

print(f"\n── Quick Correlation Check (feature vs target) ──\n")
print(f"  (sanity check — none should be suspiciously close to ±1,")
print(f"   that would suggest hidden leakage)\n")
for col in feature_cols:
    corr = ml_df[col].corr(ml_df[target_col])
    print(f"  {col:<18} r = {corr:+.3f}")

print(f"\n── What Day 25 Will Do ──\n")
print(f"  Train a Logistic Regression model on day24_train.csv")
print(f"  to predict 'sla_breach_new' from these 6 features.")
print(f"  Then evaluate on day24_test.csv using accuracy,")
print(f"  precision, recall, and a confusion matrix.")

print(f"\n{'='*65}")
print(f"  Day 24 complete.")
print(f"{'='*65}\n")