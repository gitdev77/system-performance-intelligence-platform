# day38_architecture.py
# Day 38: Technical Architecture Diagram
# Generates a visual system architecture diagram

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(1, 1, figsize=(18, 12))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis("off")

# ── Color palette ──────────────────────────────────────────
COLORS = {
    "data":    "#2E86AB",   # blue
    "physics": "#A23B72",   # purple
    "ml":      "#F18F01",   # orange
    "api":     "#C73E1D",   # red
    "obs":     "#3B1F2B",   # dark
    "present": "#1D9E75",   # green
    "arrow":   "#555555",
    "bg":      "#F8F9FA",
    "title":   "#1A1A2E",
}

fig.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["bg"])

# ── Title ──────────────────────────────────────────────────
ax.text(9, 11.4,
        "System Performance Intelligence Platform",
        ha="center", va="center", fontsize=18,
        fontweight="bold", color=COLORS["title"])
ax.text(9, 11.0,
        "Technical Architecture — 45-Day Build",
        ha="center", va="center", fontsize=11,
        color="#666666")

# ── Helper functions ───────────────────────────────────────
def draw_box(ax, x, y, w, h, label, sublabel="",
             color="#2E86AB", fontsize=9):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="white",
        linewidth=2, alpha=0.92
    )
    ax.add_patch(box)
    ax.text(x, y + (0.15 if sublabel else 0),
            label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color="white")
    if sublabel:
        ax.text(x, y - 0.22, sublabel,
                ha="center", va="center",
                fontsize=7, color="white", alpha=0.85)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["arrow"],
            lw=1.5,
            connectionstyle="arc3,rad=0.0"
        )
    )

def draw_layer_label(ax, y, text, color):
    ax.text(0.3, y, text,
            ha="left", va="center",
            fontsize=8, fontweight="bold",
            color=color, rotation=90)

# ── LAYER 1 — Data Collection (y=9.5) ─────────────────────
ax.text(9, 9.9, "LAYER 1 — DATA COLLECTION",
        ha="center", fontsize=8, color=COLORS["data"],
        fontweight="bold")

draw_box(ax, 3.5, 9.3, 2.8, 0.7,
         "Workload Simulator",
         "matrix multiply A(batch×512)×B",
         COLORS["data"])
draw_box(ax, 7.0, 9.3, 2.8, 0.7,
         "psutil Monitor",
         "CPU · Memory · Latency",
         COLORS["data"])
draw_box(ax, 10.5, 9.3, 2.8, 0.7,
         "telemetry_v2.csv",
         "104 rows · 13 batch sizes",
         COLORS["data"])
draw_box(ax, 14.0, 9.3, 2.8, 0.7,
         "cliff_analysis.csv",
         "aggregated stats · cliff flags",
         COLORS["data"])

draw_arrow(ax, 4.9, 9.3, 5.6, 9.3)
draw_arrow(ax, 8.4, 9.3, 9.1, 9.3)
draw_arrow(ax, 11.9, 9.3, 12.6, 9.3)

# ── LAYER 2 — Physics + Analytics (y=7.5) ─────────────────
ax.text(9, 8.1, "LAYER 2 — PHYSICS & ANALYTICS",
        ha="center", fontsize=8, color=COLORS["physics"],
        fontweight="bold")

draw_box(ax, 2.5, 7.5, 2.6, 0.8,
         "M/M/1 Queuing",
         "r=0.974 correlation",
         COLORS["physics"])
draw_box(ax, 5.5, 7.5, 2.6, 0.8,
         "Phase Transition",
         "critical point b_c=32",
         COLORS["physics"])
draw_box(ax, 8.5, 7.5, 2.6, 0.8,
         "Roofline Model",
         "memory-bound analysis",
         COLORS["physics"])
draw_box(ax, 11.5, 7.5, 2.6, 0.8,
         "Bottleneck Engine",
         "KMeans K=3 regimes",
         COLORS["physics"])
draw_box(ax, 14.5, 7.5, 2.6, 0.8,
         "Anomaly Detection",
         "Z-score + IQR",
         COLORS["physics"])

# arrows from Layer 1 → Layer 2
for x in [3.5, 7.0, 10.5, 14.0]:
    draw_arrow(ax, x, 8.95, x, 8.1)

# ── LAYER 3 — ML Layer (y=5.8) ────────────────────────────
ax.text(9, 6.5, "LAYER 3 — ML PREDICTION",
        ha="center", fontsize=8, color=COLORS["ml"],
        fontweight="bold")

draw_box(ax, 3.5, 5.8, 2.8, 0.8,
         "Feature Engineering",
         "rho · AI · physics features",
         COLORS["ml"])
draw_box(ax, 7.0, 5.8, 2.8, 0.8,
         "Logistic Regression",
         "precision=1.0 · AUC=1.0",
         COLORS["ml"])
draw_box(ax, 10.5, 5.8, 2.8, 0.8,
         "Threshold Optimizer",
         "threshold=0.7 → FN=0",
         COLORS["ml"])
draw_box(ax, 14.0, 5.8, 2.8, 0.8,
         "MLflow Tracking",
         "9 runs · all metrics logged",
         COLORS["ml"])

draw_arrow(ax, 4.9, 5.8, 5.6, 5.8)
draw_arrow(ax, 8.4, 5.8, 9.1, 5.8)
draw_arrow(ax, 11.9, 5.8, 12.6, 5.8)
draw_arrow(ax, 9, 7.1, 9, 6.2)

# ── LAYER 4 — API + MLOps (y=4.0) ────────────────────────
ax.text(9, 4.7, "LAYER 4 — API & MLOPS",
        ha="center", fontsize=8, color=COLORS["api"],
        fontweight="bold")

draw_box(ax, 3.0, 4.0, 3.2, 0.8,
         "FastAPI",
         "/predict /diagnose /recommend",
         COLORS["api"])
draw_box(ax, 7.0, 4.0, 2.8, 0.8,
         "Docker",
         "containerized · system-perf-api",
         COLORS["api"])
draw_box(ax, 10.5, 4.0, 2.8, 0.8,
         "PostgreSQL",
         "104 rows · 8 analytics queries",
         COLORS["api"])
draw_box(ax, 14.0, 4.0, 2.8, 0.8,
         "JSON Logging",
         "JSONL · session summaries",
         COLORS["api"])

draw_arrow(ax, 7.0, 5.4, 7.0, 4.4)
draw_arrow(ax, 4.6, 4.0, 5.6, 4.0)
draw_arrow(ax, 8.4, 4.0, 9.1, 4.0)
draw_arrow(ax, 11.9, 4.0, 12.6, 4.0)

# ── LAYER 5 — Observability (y=2.3) ──────────────────────
ax.text(9, 3.0, "LAYER 5 — OBSERVABILITY & PRESENTATION",
        ha="center", fontsize=8, color=COLORS["present"],
        fontweight="bold")

draw_box(ax, 2.5, 2.3, 2.8, 0.8,
         "Prometheus",
         "scrapes :8001/metrics · 15s",
         COLORS["present"])
draw_box(ax, 6.0, 2.3, 2.8, 0.8,
         "Grafana",
         "4 live panels · real-time",
         COLORS["present"])
draw_box(ax, 9.5, 2.3, 2.8, 0.8,
         "Tableau Dashboard",
         "4 charts · SLA heatmap",
         COLORS["present"])
draw_box(ax, 13.0, 2.3, 2.8, 0.8,
         "Cost Model",
         "$356K annual saving",
         COLORS["present"])

draw_arrow(ax, 3.0, 3.6, 3.0, 2.7)
draw_arrow(ax, 3.9, 2.3, 4.6, 2.3)
draw_arrow(ax, 7.4, 2.3, 8.1, 2.3)
draw_arrow(ax, 10.9, 2.3, 11.6, 2.3)

# ── Key findings box ──────────────────────────────────────
findings = (
    "KEY FINDINGS:  "
    "Throughput→Latency r=−0.982  |  "
    "M/M/1 queuing r=0.974  |  "
    "Performance cliff at batch=192  |  "
    "ML model precision=recall=AUC=1.0  |  "
    "Annual saving $356,441"
)
ax.text(9, 0.9, findings,
        ha="center", va="center", fontsize=8,
        color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor=COLORS["title"],
                  edgecolor="none", alpha=0.9))

# ── Tech stack footer ─────────────────────────────────────
tech = ("Python · NumPy · Pandas · Scikit-learn · FastAPI · "
        "Docker · PostgreSQL · Prometheus · Grafana · MLflow · Tableau")
ax.text(9, 0.3, tech,
        ha="center", va="center", fontsize=7.5,
        color="#888888")

plt.tight_layout()
plt.savefig("day38_architecture.png",
            dpi=150, bbox_inches="tight",
            facecolor=COLORS["bg"])
print("✅ Saved → day38_architecture.png")
plt.show()