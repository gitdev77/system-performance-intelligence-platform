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
BATCH_SIZES       = [8, 32, 64, 128, 256, 512]
REPEATS           = 5
SLA_THRESHOLD_MS  = 20.0
OUTPUT_FILE       = "/Users/devansh/Desktop/optimizeation/day17_pressure.csv"

# ─────────────────────────────────────────
# WORKLOAD SIMULATOR
# (same as your existing telemetry engine)
# ─────────────────────────────────────────
def simulate_inference(batch_size):
    A = np.random.randn(batch_size, 512)
    B = np.random.randn(512, 512)
    _ = np.dot(A, B)

# ─────────────────────────────────────────
# MEMORY PRESSURE INJECTOR
#
# Allocates a large numpy array in memory
# and holds it during experiments
# This forces your system to work with
# less available RAM — simulating what
# happens when other processes are
# competing for memory in production
# ─────────────────────────────────────────
def inject_memory_pressure(mb):
    """
    Allocate mb megabytes of memory
    Returns the array (must be kept alive
    during experiment — if we delete it
    Python frees the memory immediately)
    """
    if mb == 0:
        return None
    # Each float64 = 8 bytes
    # mb * 1024 * 1024 / 8 = number of floats
    n_elements = int(mb * 1024 * 1024 / 8)
    pressure_array = np.ones(n_elements,
                             dtype=np.float64)
    return pressure_array

# ─────────────────────────────────────────
# SINGLE EXPERIMENT RUNNER
# ─────────────────────────────────────────
def run_experiment(batch_size, scenario):
    latencies = []

    # Warm up (not recorded)
    simulate_inference(batch_size)

    # 10 inference calls measured
    for _ in range(10):
        start = time.perf_counter()
        simulate_inference(batch_size)
        end   = time.perf_counter()
        latencies.append(end - start)

    latencies     = np.array(latencies)
    avg_latency   = np.mean(latencies)
    p95_latency   = np.percentile(latencies, 95)
    p99_latency   = np.percentile(latencies, 99)
    latency_std   = np.std(latencies)
    cv_latency    = latency_std / avg_latency
    throughput    = 1.0 / avg_latency
    cpu_usage     = psutil.cpu_percent(interval=0.1)
    memory_usage  = psutil.virtual_memory().percent
    sla_breach    = 1 if p95_latency * 1000 > SLA_THRESHOLD_MS else 0

    return {
        "run_id"          : str(uuid.uuid4())[:8],
        "wall_clock_time" : datetime.utcnow().isoformat(),
        "scenario"        : scenario,
        "batch_size"      : batch_size,
        "avg_latency"     : round(avg_latency,  6),
        "p95_latency"     : round(p95_latency,  6),
        "p99_latency"     : round(p99_latency,  6),
        "latency_std"     : round(latency_std,  6),
        "cv_latency"      : round(cv_latency,   4),
        "throughput"      : round(throughput,   4),
        "cpu_usage"       : cpu_usage,
        "memory_usage"    : memory_usage,
        "sla_breach"      : sla_breach,
    }

# ─────────────────────────────────────────
# MAIN — Run 3 scenarios
# ─────────────────────────────────────────
def run_all_scenarios():
    scenarios = [
        ("baseline",        0),    # no pressure
        ("medium_pressure", 500),  # 500MB allocated
        ("high_pressure",   1000), # 1GB allocated
    ]

    all_records = []

    for scenario_name, pressure_mb in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name.upper()}")
        print(f"Memory pressure: {pressure_mb}MB allocated")
        print(f"{'='*60}\n")

        # Inject memory pressure
        # (keep reference alive during experiments)
        pressure_array = inject_memory_pressure(
            pressure_mb
        )

        # Small pause to let memory settle
        time.sleep(1)

        # Report actual memory state
        mem = psutil.virtual_memory()
        print(f"System memory: {mem.percent:.1f}% used "
              f"({mem.available // (1024**2)}MB available)\n")

        total = len(BATCH_SIZES) * REPEATS
        count = 0

        for batch_size in BATCH_SIZES:
            for repeat in range(REPEATS):
                record = run_experiment(
                    batch_size, scenario_name
                )
                all_records.append(record)
                count += 1
                print(
                    f"[{scenario_name}] "
                    f"[{count}/{total}] "
                    f"batch={batch_size} | "
                    f"avg={record['avg_latency']*1000:.2f}ms | "
                    f"p95={record['p95_latency']*1000:.2f}ms | "
                    f"mem={record['memory_usage']:.1f}% | "
                    f"sla={'BREACH' if record['sla_breach'] else 'OK'}"
                )

        # Release memory pressure
        # (Python garbage collects the array)
        del pressure_array
        pressure_array = None
        time.sleep(1)

    # Save dataset
    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Saved → {OUTPUT_FILE}")
    print(f"Rows: {len(df)} | "
          f"Scenarios: {df['scenario'].nunique()}")
    return df

# ─────────────────────────────────────────
# ANALYSIS — Compare 3 scenarios
# ─────────────────────────────────────────
def analyze_scenarios(df):
    df["avg_latency_ms"] = df["avg_latency"] * 1000
    df["p95_latency_ms"] = df["p95_latency"] * 1000
    df["p99_latency_ms"] = df["p99_latency"] * 1000

    agg = df.groupby(
        ["scenario", "batch_size"]
    ).agg(
        avg_latency_ms = ("avg_latency_ms", "mean"),
        p95_latency_ms = ("p95_latency_ms", "mean"),
        p99_latency_ms = ("p99_latency_ms", "mean"),
        throughput     = ("throughput",     "mean"),
        memory_usage   = ("memory_usage",   "mean"),
        breach_rate    = ("sla_breach",     "mean"),
    ).reset_index()

    agg["breach_rate_pct"] = (
        agg["breach_rate"] * 100
    ).round(1)

    return agg

# ─────────────────────────────────────────
# PLOT — 4 panel comparison dashboard
# ─────────────────────────────────────────
def plot_comparison(agg):
    fig, axes = plt.subplots(4, 1, figsize=(13, 18))
    fig.suptitle(
        "Memory Pressure Simulation — "
        "3 Scenario Comparison\n"
        "System Performance Intelligence Platform",
        fontsize=14, fontweight="bold", y=0.98
    )

    scenario_colors = {
        "baseline"       : "#1D9E75",
        "medium_pressure": "#EF9F27",
        "high_pressure"  : "#D85A30",
    }
    scenario_labels = {
        "baseline"       : "Baseline (no pressure)",
        "medium_pressure": "Medium pressure (500MB)",
        "high_pressure"  : "High pressure (1GB)",
    }

    scenarios = [
        "baseline",
        "medium_pressure",
        "high_pressure"
    ]

    # ── PLOT 1 — Avg latency comparison ──
    ax1 = axes[0]
    for scenario in scenarios:
        data = agg[agg["scenario"] == scenario]
        ax1.plot(
            data["batch_size"],
            data["avg_latency_ms"],
            color=scenario_colors[scenario],
            linewidth=2.5, marker="o",
            markersize=6,
            label=scenario_labels[scenario]
        )
    ax1.axhline(
        y=SLA_THRESHOLD_MS,
        color="red", linewidth=1.5,
        linestyle="-.",
        label="SLA threshold (20ms)"
    )
    ax1.set_title(
        "Avg Latency — Baseline vs Memory Pressure\n"
        "(does memory pressure shift the latency curve?)",
        fontsize=11, fontweight="bold"
    )
    ax1.set_ylabel("Avg Latency (ms)")
    ax1.set_xticks(BATCH_SIZES)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── PLOT 2 — p99 tail latency ──
    ax2 = axes[1]
    for scenario in scenarios:
        data = agg[agg["scenario"] == scenario]
        ax2.plot(
            data["batch_size"],
            data["p99_latency_ms"],
            color=scenario_colors[scenario],
            linewidth=2.5, marker="^",
            markersize=6,
            label=scenario_labels[scenario]
        )
    ax2.axhline(
        y=SLA_THRESHOLD_MS,
        color="red", linewidth=1.5,
        linestyle="-.",
        label="SLA threshold (20ms)"
    )
    ax2.set_title(
        "p99 Tail Latency — Does Memory Pressure "
        "Worsen Your Worst Requests?",
        fontsize=11, fontweight="bold"
    )
    ax2.set_ylabel("p99 Latency (ms)")
    ax2.set_xticks(BATCH_SIZES)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── PLOT 3 — Throughput comparison ──
    ax3 = axes[2]
    for scenario in scenarios:
        data = agg[agg["scenario"] == scenario]
        ax3.plot(
            data["batch_size"],
            data["throughput"],
            color=scenario_colors[scenario],
            linewidth=2.5, marker="s",
            markersize=6,
            label=scenario_labels[scenario]
        )
    ax3.set_title(
        "Throughput — Does Memory Pressure "
        "Collapse Throughput Faster?",
        fontsize=11, fontweight="bold"
    )
    ax3.set_ylabel("Throughput (req/s)")
    ax3.set_xticks(BATCH_SIZES)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── PLOT 4 — SLA breach rate ──
    ax4 = axes[3]
    width = 15
    x     = np.array(BATCH_SIZES)

    for i, scenario in enumerate(scenarios):
        data = agg[agg["scenario"] == scenario]
        offset = (i - 1) * width
        ax4.bar(
            x + offset,
            data["breach_rate_pct"].values,
            width=width,
            color=scenario_colors[scenario],
            alpha=0.85,
            label=scenario_labels[scenario]
        )

    ax4.axhline(
        y=50, color="#D85A30",
        linewidth=2, linestyle="--",
        label="Critical threshold (50%)"
    )
    ax4.set_title(
        "SLA Breach Rate — Does Memory Pressure "
        "Cause More Breaches?",
        fontsize=11, fontweight="bold"
    )
    ax4.set_ylabel("Breach Rate (%)")
    ax4.set_xlabel("Batch size")
    ax4.set_xticks(x)
    ax4.set_xticklabels(BATCH_SIZES)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        "/Users/devansh/Desktop/optimizeation/"
        "day17_pressure_comparison.png",
        dpi=150, bbox_inches="tight"
    )
    print("✅ Saved → day17_pressure_comparison.png")

# ─────────────────────────────────────────
# PRINT FINDINGS
# ─────────────────────────────────────────
def print_findings(agg):
    print("\n── Day 17 Findings ──\n")

    for scenario in [
        "baseline",
        "medium_pressure",
        "high_pressure"
    ]:
        data = agg[agg["scenario"] == scenario]
        avg_breach = data["breach_rate_pct"].mean()
        max_p99    = data["p99_latency_ms"].max()
        min_tput   = data["throughput"].min()

        print(f"  {scenario.upper():<20} | "
              f"avg breach={avg_breach:.1f}% | "
              f"max p99={max_p99:.2f}ms | "
              f"min throughput={min_tput:.1f} req/s")

    print("\n── Memory Impact Analysis ──\n")

    base = agg[agg["scenario"] == "baseline"]
    high = agg[agg["scenario"] == "high_pressure"]

    for batch in BATCH_SIZES:
        b_lat = base[
            base["batch_size"] == batch
        ]["avg_latency_ms"].values
        h_lat = high[
            high["batch_size"] == batch
        ]["avg_latency_ms"].values

        if len(b_lat) > 0 and len(h_lat) > 0:
            delta = h_lat[0] - b_lat[0]
            pct   = (delta / b_lat[0]) * 100
            impact = "↑ worse" if delta > 0 \
                else "↓ better"
            print(
                f"  batch={batch:<6} | "
                f"baseline={b_lat[0]:.2f}ms | "
                f"high_pressure={h_lat[0]:.2f}ms | "
                f"delta={delta:+.2f}ms "
                f"({pct:+.1f}%) {impact}"
            )

# ─────────────────────────────────────────
# RUN EVERYTHING
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("Day 17 — Memory Pressure Simulation")
    print("3 scenarios × 6 batch sizes × "
          f"5 repeats = "
          f"{3 * len(BATCH_SIZES) * REPEATS} experiments\n")

    df  = run_all_scenarios()
    agg = analyze_scenarios(df)
    plot_comparison(agg)
    print_findings(agg)