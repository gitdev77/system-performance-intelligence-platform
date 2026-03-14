import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("/Users/devansh/Desktop/optimizeation/telemetry_v2.csv")

df["cpu_usage"] = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage"] = df["cpu_usage"].ffill()

# Convert to ms
df["avg_latency_ms"] = df["avg_latency"] * 1000

# ─────────────────────────────────────────
# FIGURE — 2x2 scatter grid
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Performance Relationships — Scatter Analysis\nSystem Performance Intelligence Platform",
    fontsize=14, fontweight="bold", y=0.98
)

# Color points by batch size — larger batch = darker
colors = df["batch_size"]

# ─────────────────────────────────────────
# PLOT 1 — CPU vs Avg Latency
# ─────────────────────────────────────────
ax1 = axes[0, 0]
sc1 = ax1.scatter(df["cpu_usage"], df["avg_latency_ms"],
                  c=colors, cmap="viridis", alpha=0.6, s=40)

# Regression line
slope, intercept, r, p, se = stats.linregress(
    df["cpu_usage"], df["avg_latency_ms"]
)
x_line = np.linspace(df["cpu_usage"].min(), df["cpu_usage"].max(), 100)
ax1.plot(x_line, slope * x_line + intercept,
         color="red", linewidth=2, linestyle="--",
         label=f"r={r:.3f}, p={p:.3f}")

ax1.set_xlabel("CPU Usage (%)")
ax1.set_ylabel("Avg Latency (ms)")
ax1.set_title("CPU vs Latency\n(weak relationship expected)", fontsize=10,
              fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
plt.colorbar(sc1, ax=ax1, label="Batch size")

# ─────────────────────────────────────────
# PLOT 2 — Batch Size vs Avg Latency
# ─────────────────────────────────────────
ax2 = axes[0, 1]
sc2 = ax2.scatter(df["batch_size"], df["avg_latency_ms"],
                  c=colors, cmap="viridis", alpha=0.6, s=40)

slope2, intercept2, r2, p2, se2 = stats.linregress(
    df["batch_size"], df["avg_latency_ms"]
)
x_line2 = np.linspace(df["batch_size"].min(), df["batch_size"].max(), 100)
ax2.plot(x_line2, slope2 * x_line2 + intercept2,
         color="red", linewidth=2, linestyle="--",
         label=f"r={r2:.3f}, p={p2:.3f}")

ax2.set_xlabel("Batch Size")
ax2.set_ylabel("Avg Latency (ms)")
ax2.set_title("Batch Size vs Latency\n(moderate relationship)", fontsize=10,
              fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.colorbar(sc2, ax=ax2, label="Batch size")

# ─────────────────────────────────────────
# PLOT 3 — Memory vs Avg Latency
# ─────────────────────────────────────────
ax3 = axes[1, 0]
sc3 = ax3.scatter(df["memory_usage"], df["avg_latency_ms"],
                  c=colors, cmap="viridis", alpha=0.6, s=40)

slope3, intercept3, r3, p3, se3 = stats.linregress(
    df["memory_usage"], df["avg_latency_ms"]
)
x_line3 = np.linspace(df["memory_usage"].min(), df["memory_usage"].max(), 100)
ax3.plot(x_line3, slope3 * x_line3 + intercept3,
         color="red", linewidth=2, linestyle="--",
         label=f"r={r3:.3f}, p={p3:.3f}")

ax3.set_xlabel("Memory Usage (%)")
ax3.set_ylabel("Avg Latency (ms)")
ax3.set_title("Memory vs Latency\n(weak relationship expected)", fontsize=10,
              fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
plt.colorbar(sc3, ax=ax3, label="Batch size")

# ─────────────────────────────────────────
# PLOT 4 — Throughput vs Avg Latency
# ─────────────────────────────────────────
ax4 = axes[1, 1]
sc4 = ax4.scatter(df["throughput"], df["avg_latency_ms"],
                  c=colors, cmap="viridis", alpha=0.6, s=40)

slope4, intercept4, r4, p4, se4 = stats.linregress(
    df["throughput"], df["avg_latency_ms"]
)
x_line4 = np.linspace(df["throughput"].min(), df["throughput"].max(), 100)
ax4.plot(x_line4, slope4 * x_line4 + intercept4,
         color="red", linewidth=2, linestyle="--",
         label=f"r={r4:.3f}, p={p4:.3f}")

ax4.set_xlabel("Throughput (req/s)")
ax4.set_ylabel("Avg Latency (ms)")
ax4.set_title("Throughput vs Latency\n(strong inverse relationship)", fontsize=10,
              fontweight="bold")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
plt.colorbar(sc4, ax=ax4, label="Batch size")

# ─────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/Users/devansh/Desktop/optimizeation/scatter_analysis.png",
            dpi=150, bbox_inches="tight")
print("✅ Saved → scatter_analysis.png")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
print("\n── Scatter Analysis Findings ──\n")
findings = [
    ("CPU vs Latency",        r,  p),
    ("Batch Size vs Latency", r2, p2),
    ("Memory vs Latency",     r3, p3),
    ("Throughput vs Latency", r4, p4),
]
for name, r_val, p_val in findings:
    sig = "SIGNIFICANT" if p_val < 0.05 else "NOT significant"
    print(f"  {name:<25} r={r_val:+.3f}  p={p_val:.4f}  [{sig}]")