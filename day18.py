import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import psutil
import time
import uuid
from datetime import datetime

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SLA_THRESHOLD_MS = 20.0
OUTPUT_FILE = "/Users/devansh/Desktop/optimizeation/day18_regimes.csv"

# ─────────────────────────────────────────
# REGIME DEFINITIONS
#
# Light load = small batches, few repeats
# simulates: low traffic, off-peak hours
#
# Heavy load = large batches, many repeats
# simulates: peak traffic, high demand
# ─────────────────────────────────────────
REGIMES = {
    "light_load": {
        "batch_sizes" : [8, 16, 32, 48, 64],
        "repeats"     : 10,
        "description" : "Small batches, low frequency"
    },
    "heavy_load": {
        "batch_sizes" : [128, 192, 256, 320, 384, 448, 512],
        "repeats"     : 10,
        "description" : "Large batches, high frequency"
    }
}

# ─────────────────────────────────────────
# WORKLOAD SIMULATOR
# ─────────────────────────────────────────
def simulate_inference(batch_size):
    A = np.random.randn(batch_size, 512)
    B = np.random.randn(512, 512)
    _ = np.dot(A, B)

# ─────────────────────────────────────────
# SINGLE EXPERIMENT
# ─────────────────────────────────────────
def run_experiment(batch_size, regime):
    latencies = []

    simulate_inference(batch_size)  # warmup

    for _ in range(10):
        start = time.perf_counter()
        simulate_inference(batch_size)
        end   = time.perf_counter()
        latencies.append(end - start)

    latencies    = np.array(latencies)
    avg_latency  = np.mean(latencies)
    p95_latency  = np.percentile(latencies, 95)
    p99_latency  = np.percentile(latencies, 99)
    latency_std  = np.std(latencies)
    cv_latency   = latency_std / avg_latency
    throughput   = 1.0 / avg_latency
    cpu_usage    = psutil.cpu_percent(interval=0.1)
    memory_usage = psutil.virtual_memory().percent
    sla_breach   = 1 if p95_latency * 1000 > SLA_THRESHOLD_MS else 0

    return {
        "run_id"         : str(uuid.uuid4())[:8],
        "wall_clock_time": datetime.utcnow().isoformat(),
        "regime"         : regime,
        "batch_size"     : batch_size,
        "avg_latency"    : round(avg_latency,  6),
        "p95_latency"    : round(p95_latency,  6),
        "p99_latency"    : round(p99_latency,  6),
        "latency_std"    : round(latency_std,  6),
        "cv_latency"     : round(cv_latency,   4),
        "throughput"     : round(throughput,   4),
        "cpu_usage"      : cpu_usage,
        "memory_usage"   : memory_usage,
        "sla_breach"     : sla_breach,
    }

# ─────────────────────────────────────────
# RUN ALL REGIMES
# ─────────────────────────────────────────
def run_all_regimes():
    all_records = []

    for regime_name, config in REGIMES.items():
        print(f"\n{'='*60}")
        print(f"REGIME: {regime_name.upper()}")
        print(f"Description: {config['description']}")
        print(f"Batch sizes: {config['batch_sizes']}")
        print(f"Repeats: {config['repeats']}")
        print(f"{'='*60}\n")

        total = len(config["batch_sizes"]) * config["repeats"]
        count = 0

        for batch_size in config["batch_sizes"]:
            for repeat in range(config["repeats"]):
                record = run_experiment(batch_size, regime_name)
                all_records.append(record)
                count += 1
                print(
                    f"[{regime_name}] "
                    f"[{count}/{total}] "
                    f"batch={batch_size} | "
                    f"avg={record['avg_latency']*1000:.2f}ms | "
                    f"p95={record['p95_latency']*1000:.2f}ms | "
                    f"cpu={record['cpu_usage']:.1f}% | "
                    f"sla={'BREACH' if record['sla_breach'] else 'OK'}"
                )

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved → {OUTPUT_FILE}")
    print(f"Rows: {len(df)}")
    return df

# ─────────────────────────────────────────
# ANALYZE REGIMES
# ─────────────────────────────────────────
def analyze_regimes(df):
    df["avg_latency_ms"] = df["avg_latency"] * 1000
    df["p95_latency_ms"] = df["p95_latency"] * 1000
    df["p99_latency_ms"] = df["p99_latency"] * 1000

    # Per batch size aggregation
    agg = df.groupby(["regime", "batch_size"]).agg(
        avg_latency_ms = ("avg_latency_ms", "mean"),
        p95_latency_ms = ("p95_latency_ms", "mean"),
        p99_latency_ms = ("p99_latency_ms", "mean"),
        throughput     = ("throughput",     "mean"),
        cpu_usage      = ("cpu_usage",      "mean"),
        cv_latency     = ("cv_latency",     "mean"),
        breach_rate    = ("sla_breach",     "mean"),
    ).reset_index()

    agg["breach_rate_pct"] = (
        agg["breach_rate"] * 100
    ).round(1)

    # Regime summary
    summary = df.groupby("regime").agg(
        avg_latency_ms  = ("avg_latency_ms",  "mean"),
        p95_latency_ms  = ("p95_latency_ms",  "mean"),
        p99_latency_ms  = ("p99_latency_ms",  "mean"),
        throughput      = ("throughput",      "mean"),
        breach_rate_pct = ("sla_breach",      lambda x: x.mean() * 100),
        cv_latency      = ("cv_latency",      "mean"),
    ).round(2).reset_index()

    return agg, summary

# ─────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────
def plot_regimes(agg, summary):
    fig, axes = plt.subplots(4, 1, figsize=(13, 18))
    fig.suptitle(
        "Light vs Heavy Load Regime Comparison\n"
        "System Performance Intelligence Platform",
        fontsize=14, fontweight="bold", y=0.98
    )

    regime_colors = {
        "light_load": "#1D9E75",
        "heavy_load": "#D85A30",
    }
    regime_labels = {
        "light_load": "Light load (batch 8-64)",
        "heavy_load": "Heavy load (batch 128-512)",
    }

    # ── PLOT 1 — Latency comparison ──
    ax1 = axes[0]

    for regime in ["light_load", "heavy_load"]:
        data = agg[agg["regime"] == regime]
        ax1.plot(
            data["batch_size"],
            data["avg_latency_ms"],
            color=regime_colors[regime],
            linewidth=2.5, marker="o",
            markersize=6,
            label=f"{regime_labels[regime]} — avg"
        )
        ax1.fill_between(
            data["batch_size"],
            data["avg_latency_ms"],
            data["p99_latency_ms"],
            alpha=0.15,
            color=regime_colors[regime],
            label=f"{regime_labels[regime]} — p99 band"
        )

    ax1.axhline(
        y=SLA_THRESHOLD_MS,
        color="red", linewidth=1.5,
        linestyle="-.",
        label="SLA threshold (20ms)"
    )

    # Regime boundary line
    ax1.axvline(
        x=96, color="gray",
        linewidth=2, linestyle="--",
        alpha=0.7, label="Regime boundary (~batch=96)"
    )

    ax1.set_title(
        "Avg Latency + p99 Band — Light vs Heavy Load\n"
        "(shaded area = gap between avg and p99)",
        fontsize=11, fontweight="bold"
    )
    ax1.set_ylabel("Latency (ms)")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── PLOT 2 — Throughput ──
    ax2 = axes[1]

    for regime in ["light_load", "heavy_load"]:
        data = agg[agg["regime"] == regime]
        ax2.plot(
            data["batch_size"],
            data["throughput"],
            color=regime_colors[regime],
            linewidth=2.5, marker="s",
            markersize=6,
            label=regime_labels[regime]
        )

    ax2.set_title(
        "Throughput — Light vs Heavy Load\n"
        "(how many requests per second each regime handles)",
        fontsize=11, fontweight="bold"
    )
    ax2.set_ylabel("Throughput (req/s)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── PLOT 3 — CV Latency (consistency) ──
    ax3 = axes[2]

    for regime in ["light_load", "heavy_load"]:
        data = agg[agg["regime"] == regime]
        ax3.plot(
            data["batch_size"],
            data["cv_latency"],
            color=regime_colors[regime],
            linewidth=2.5, marker="^",
            markersize=6,
            label=regime_labels[regime]
        )

    ax3.set_title(
        "CV Latency — Consistency Per Regime\n"
        "(higher = more inconsistent behavior)",
        fontsize=11, fontweight="bold"
    )
    ax3.set_ylabel("Coefficient of Variation")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── PLOT 4 — Regime summary bars ──
    ax4 = axes[3]

    metrics = [
        "avg_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "breach_rate_pct"
    ]
    labels = [
        "Avg Latency\n(ms)",
        "p95 Latency\n(ms)",
        "p99 Latency\n(ms)",
        "Breach Rate\n(%)"
    ]

    x     = np.arange(len(metrics))
    width = 0.35

    for i, regime in enumerate(["light_load", "heavy_load"]):
        row = summary[summary["regime"] == regime].iloc[0]
        values = [row[m] for m in metrics]
        bars = ax4.bar(
            x + i * width,
            values,
            width=width,
            color=regime_colors[regime],
            alpha=0.85,
            label=regime_labels[regime]
        )
        for bar, val in zip(bars, values):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold"
            )

    ax4.set_title(
        "Regime Summary — Key Metrics Comparison\n"
        "(direct comparison of light vs heavy load)",
        fontsize=11, fontweight="bold"
    )
    ax4.set_xticks(x + width / 2)
    ax4.set_xticklabels(labels, fontsize=9)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        "/Users/devansh/Desktop/optimizeation/day18_regimes.png",
        dpi=150, bbox_inches="tight"
    )
    print("✅ Saved → day18_regimes.png")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
def print_findings(summary, agg):
    print("\n── Day 18 Findings ──\n")
    print(summary.to_string(index=False))

    print("\n── Regime Gap Analysis ──\n")
    light = summary[
        summary["regime"] == "light_load"
    ].iloc[0]
    heavy = summary[
        summary["regime"] == "heavy_load"
    ].iloc[0]

    metrics = [
        ("avg_latency_ms",  "Avg latency"),
        ("p99_latency_ms",  "p99 latency"),
        ("throughput",      "Throughput"),
        ("breach_rate_pct", "Breach rate"),
    ]

    for col, name in metrics:
        gap = heavy[col] - light[col]
        pct = (gap / light[col]) * 100
        direction = "worse" if gap > 0 else "better"
        if col == "throughput":
            direction = "worse" if gap < 0 else "better"
        print(
            f"  {name:<15} | "
            f"light={light[col]:.2f} | "
            f"heavy={heavy[col]:.2f} | "
            f"gap={gap:+.2f} ({pct:+.1f}%) "
            f"→ heavy is {direction}"
        )

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Day 18 — Light vs Heavy Load Comparison")
    print("Two regimes × experiments running...\n")

    df          = run_all_regimes()
    agg, summary = analyze_regimes(df)
    plot_regimes(agg, summary)
    print_findings(summary, agg)