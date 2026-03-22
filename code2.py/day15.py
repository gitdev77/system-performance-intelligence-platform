import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("/Users/devansh/Desktop/optimizeation/telemetry_v2.csv")

df["cpu_usage"]      = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage"]      = df["cpu_usage"].ffill()
df["avg_latency_ms"] = df["avg_latency"] * 1000
df["p95_latency_ms"] = df["p95_latency"] * 1000

# ─────────────────────────────────────────
# STEP 1 — Define features and target
# Features = variables we think might
# cause latency
# Target = what we are trying to explain
# (avg latency)
# ─────────────────────────────────────────
features = [
    "batch_size",     # size of input batch
    "cpu_usage",      # CPU utilization %
    "memory_usage",   # RAM utilization %
    "throughput",     # requests per second
    "cv_latency",     # coefficient of variation
                      # (consistency measure)
]

target = "avg_latency_ms"

# Drop rows with any missing values
data = df[features + [target]].dropna()

X = data[features].values
y = data[target].values

# ─────────────────────────────────────────
# STEP 2 — Standardize features
# StandardScaler transforms each feature
# to have mean=0 and std=1
# This is critical — without it a feature
# measured in large numbers (batch_size
# goes up to 512) would dominate features
# measured in small numbers (cv_latency
# is 0.3-0.9) just because of scale
# ─────────────────────────────────────────
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────
# STEP 3 — Multiple linear regression
# Fits a line through ALL variables
# simultaneously
# coefficient (weight given to each
# variable) tells us its contribution
# ─────────────────────────────────────────
model = LinearRegression()
model.fit(X_scaled, y)

y_pred  = model.predict(X_scaled)
r2      = r2_score(y, y_pred)

# ─────────────────────────────────────────
# STEP 4 — Extract coefficients
# (how much each variable contributes
# to latency when all others are
# held constant)
# Positive coefficient = higher value
# of this variable → higher latency
# Negative coefficient = higher value
# → lower latency
# ─────────────────────────────────────────
coef_df = pd.DataFrame({
    "feature"    : features,
    "coefficient": model.coef_,
    "abs_coef"   : np.abs(model.coef_)
}).sort_values("abs_coef", ascending=False)

# ─────────────────────────────────────────
# STEP 5 — Variance explained per feature
# What % of latency variance does each
# feature explain on its own?
# ─────────────────────────────────────────
variance_explained = {}
for i, feat in enumerate(features):
    X_single = X_scaled[:, i].reshape(-1, 1)
    m = LinearRegression()
    m.fit(X_single, y)
    r2_single = r2_score(y, m.predict(X_single))
    variance_explained[feat] = max(r2_single, 0) * 100

var_df = pd.DataFrame({
    "feature" : list(variance_explained.keys()),
    "variance_pct" : list(variance_explained.values())
}).sort_values("variance_pct", ascending=False)

# ─────────────────────────────────────────
# STEP 6 — Partial correlation
# Correlation of each variable with
# latency AFTER removing effect of
# all other variables
# ─────────────────────────────────────────
partial_corr = {}
for feat in features:
    other_feats = [f for f in features if f != feat]
    X_others    = data[other_feats].values
    X_others_sc = StandardScaler().fit_transform(X_others)

    # Residuals of feat after removing
    # effect of other features
    m1 = LinearRegression()
    m1.fit(X_others_sc, X_scaled[:, features.index(feat)])
    resid_feat = X_scaled[:, features.index(feat)] - m1.predict(X_others_sc)

    # Residuals of target after removing
    # effect of other features
    m2 = LinearRegression()
    m2.fit(X_others_sc, y)
    resid_y = y - m2.predict(X_others_sc)

    partial_corr[feat] = np.corrcoef(resid_feat, resid_y)[0, 1]

pc_df = pd.DataFrame({
    "feature"      : list(partial_corr.keys()),
    "partial_corr" : list(partial_corr.values())
}).sort_values("partial_corr", key=abs, ascending=False)

# ─────────────────────────────────────────
# BOTTLENECK RANKING
# Combine coefficient + variance +
# partial correlation into one rank
# ─────────────────────────────────────────
rank_df = coef_df.merge(var_df, on="feature")
rank_df = rank_df.merge(pc_df, on="feature")
rank_df["bottleneck_score"] = (
    0.4 * (rank_df["abs_coef"] / rank_df["abs_coef"].max()) +
    0.4 * (rank_df["variance_pct"] / rank_df["variance_pct"].max()) +
    0.2 * (rank_df["partial_corr"].abs() /
           rank_df["partial_corr"].abs().max())
).round(3)
rank_df = rank_df.sort_values(
    "bottleneck_score", ascending=False
).reset_index(drop=True)
rank_df["rank"] = rank_df.index + 1

# ─────────────────────────────────────────
# PLOT — 4 panel bottleneck dashboard
# ─────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(13, 18))
fig.suptitle(
    "Multi-Variable Bottleneck Analysis\n"
    "System Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

feat_colors = {
    "batch_size"  : "#534AB7",
    "cpu_usage"   : "#D85A30",
    "memory_usage": "#EF9F27",
    "throughput"  : "#1D9E75",
    "cv_latency"  : "#0F6E56"
}
bar_colors = [feat_colors[f] for f in coef_df["feature"]]

# ── PLOT 1 — Regression coefficients ──
ax1 = axes[0]

bars = ax1.barh(
    coef_df["feature"],
    coef_df["coefficient"],
    color=[feat_colors[f] for f in coef_df["feature"]],
    alpha=0.85
)
ax1.axvline(x=0, color="black", linewidth=1)

for bar, val in zip(bars, coef_df["coefficient"]):
    xpos = bar.get_width() + 0.05 \
        if bar.get_width() >= 0 \
        else bar.get_width() - 0.05
    ax1.text(
        xpos, bar.get_y() + bar.get_height()/2,
        f"{val:+.3f}",
        va="center", fontsize=9, fontweight="bold"
    )

ax1.set_title(
    f"Regression Coefficients — Each Variable's Contribution to Latency\n"
    f"(Model R² = {r2:.3f} — explains {r2*100:.1f}% of latency variance)",
    fontsize=11, fontweight="bold"
)
ax1.set_xlabel("Coefficient (standardized)")
ax1.grid(True, alpha=0.3, axis="x")

# ── PLOT 2 — Variance explained ──
ax2 = axes[1]

bars2 = ax2.barh(
    var_df["feature"],
    var_df["variance_pct"],
    color=[feat_colors[f] for f in var_df["feature"]],
    alpha=0.85
)

for bar, val in zip(bars2, var_df["variance_pct"]):
    ax2.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height()/2,
        f"{val:.1f}%",
        va="center", fontsize=9, fontweight="bold"
    )

ax2.set_title(
    "Variance Explained — What % of Latency Does Each Variable Explain Alone?",
    fontsize=11, fontweight="bold"
)
ax2.set_xlabel("Variance Explained (%)")
ax2.grid(True, alpha=0.3, axis="x")

# ── PLOT 3 — Partial correlation ──
ax3 = axes[2]

pc_colors = [
    "#1D9E75" if v > 0 else "#D85A30"
    for v in pc_df["partial_corr"]
]
bars3 = ax3.barh(
    pc_df["feature"],
    pc_df["partial_corr"],
    color=[feat_colors[f] for f in pc_df["feature"]],
    alpha=0.85
)
ax3.axvline(x=0, color="black", linewidth=1)

for bar, val in zip(bars3, pc_df["partial_corr"]):
    xpos = bar.get_width() + 0.01 \
        if bar.get_width() >= 0 \
        else bar.get_width() - 0.01
    ax3.text(
        xpos, bar.get_y() + bar.get_height()/2,
        f"{val:+.3f}",
        va="center", fontsize=9, fontweight="bold"
    )

ax3.set_title(
    "Partial Correlation — Each Variable's Unique Effect on Latency\n"
    "(after removing the influence of all other variables)",
    fontsize=11, fontweight="bold"
)
ax3.set_xlabel("Partial Correlation")
ax3.grid(True, alpha=0.3, axis="x")

# ── PLOT 4 — Bottleneck ranking ──
ax4 = axes[3]

rank_colors = [feat_colors[f] for f in rank_df["feature"]]
bars4 = ax4.barh(
    rank_df["feature"],
    rank_df["bottleneck_score"],
    color=rank_colors, alpha=0.85
)

for bar, val, rnk in zip(
    bars4,
    rank_df["bottleneck_score"],
    rank_df["rank"]
):
    ax4.text(
        bar.get_width() + 0.01,
        bar.get_y() + bar.get_height()/2,
        f"#{rnk}  score={val:.3f}",
        va="center", fontsize=9, fontweight="bold"
    )

ax4.set_title(
    "Bottleneck Ranking — Combined Score From All 3 Analyses\n"
    "(#1 = biggest contributor to latency degradation)",
    fontsize=11, fontweight="bold"
)
ax4.set_xlabel("Bottleneck Score (0-1)")
ax4.grid(True, alpha=0.3, axis="x")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    "/Users/devansh/Desktop/optimizeation/day15_bottleneck.png",
    dpi=150, bbox_inches="tight"
)
print("✅ Saved → day15_bottleneck.png")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print(f"\n── Day 15 Findings ──\n")
print(f"Model R² score: {r2:.3f}")
print(f"(model explains {r2*100:.1f}% of latency variance)\n")

print("── Bottleneck Ranking ──\n")
for _, row in rank_df.iterrows():
    direction = "increases" if row["coefficient"] > 0 \
        else "decreases"
    print(
        f"  #{int(row['rank'])} {row['feature']:<15} "
        f"score={row['bottleneck_score']:.3f} | "
        f"variance={row['variance_pct']:.1f}% | "
        f"partial_r={row['partial_corr']:+.3f} | "
        f"{direction} latency"
    )

print(f"\n── Primary Bottleneck ──\n")
top = rank_df.iloc[0]
print(f"  {top['feature']} is the #1 contributor to latency")
print(f"  It alone explains {top['variance_pct']:.1f}% of latency variance")
print(f"  Partial correlation: {top['partial_corr']:+.3f}")