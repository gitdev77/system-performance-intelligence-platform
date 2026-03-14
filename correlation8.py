import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("/Users/devansh/Desktop/optimizeation/telemetry_v2.csv")

# Fix CPU psutil bug on Apple Silicon
df["cpu_usage"] = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage"] = df["cpu_usage"].ffill()

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────
# CORRELATION MATRIX
# ─────────────────────────────────────────
metrics = [
    "batch_size", "avg_latency", "p95_latency",
    "p99_latency", "latency_std", "cv_latency",
    "throughput", "cpu_usage", "memory_usage"
]

corr = df[metrics].corr(method="pearson")

# ─────────────────────────────────────────
# PLOT — heatmap
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)

ax.set_xticks(range(len(metrics)))
ax.set_yticks(range(len(metrics)))
ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(metrics, fontsize=9)

for i in range(len(metrics)):
    for j in range(len(metrics)):
        val = corr.iloc[i, j]
        color = "black" if abs(val) < 0.7 else "white"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8, color=color, fontweight="bold")

ax.set_title("Pearson Correlation Matrix\nSystem Performance Metrics",
             fontsize=13, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig("/Users/devansh/Desktop/optimizeation/correlation_heatmap.png",
            dpi=150, bbox_inches="tight")
print("✅ Saved → correlation_heatmap.png")

# ─────────────────────────────────────────
# PRINT KEY CORRELATIONS WITH AVG LATENCY
# ─────────────────────────────────────────
print("\n── Correlations with avg_latency ──\n")
lat_corr = corr["avg_latency"].drop("avg_latency").sort_values(
    key=abs, ascending=False
)
for metric, val in lat_corr.items():
    strength = (
        "STRONG"   if abs(val) > 0.7 else
        "MODERATE" if abs(val) > 0.4 else
        "WEAK"
    )
    direction = "positive" if val > 0 else "negative"
    print(f"  {metric:<20} r={val:+.3f}  [{strength} {direction}]")

print("\n── Correlations with cpu_usage ──\n")
cpu_corr = corr["cpu_usage"].drop("cpu_usage").sort_values(
    key=abs, ascending=False
)
for metric, val in cpu_corr.items():
    strength = (
        "STRONG"   if abs(val) > 0.7 else
        "MODERATE" if abs(val) > 0.4 else
        "WEAK"
    )
    direction = "positive" if val > 0 else "negative"
    print(f"  {metric:<20} r={val:+.3f}  [{strength} {direction}]")

print("\nDone!")