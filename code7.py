import psutil
import numpy as np
import pandas as pd
import time
import uuid
from datetime import datetime

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BATCH_SIZES = [8, 16, 32, 48, 64, 96, 128, 192, 256, 320, 384, 448, 512]
REPEATS_PER_BATCH = 8          # 8 runs per batch size = 104 rows total
SLA_THRESHOLD_P95 = 0.15       # 150ms SLA on p95
OUTPUT_FILE = "telemetry_v2.csv"

# ─────────────────────────────────────────
# WORKLOAD SIMULATOR
# ─────────────────────────────────────────
def simulate_inference(batch_size):
    """
    Simulates PyTorch-style matrix inference.
    Intentionally stresses CPU to create a real performance cliff.
    """
    A = np.random.randn(batch_size, 512)
    B = np.random.randn(512, 512)
    _ = np.dot(A, B)  # Matrix multiply — scales non-linearly with batch

# ─────────────────────────────────────────
# SINGLE EXPERIMENT RUNNER
# ─────────────────────────────────────────
def run_experiment(batch_size, repeat_index):
    latencies = []

    # Warm-up (not recorded)
    simulate_inference(batch_size)

    # Measure 10 inference calls
    for _ in range(10):
        start = time.perf_counter()
        simulate_inference(batch_size)
        end = time.perf_counter()
        latencies.append(end - start)

    latencies = np.array(latencies)

    # Metrics
    avg_latency   = np.mean(latencies)
    p95_latency   = np.percentile(latencies, 95)
    p99_latency   = np.percentile(latencies, 99)
    latency_std   = np.std(latencies)
    cv_latency    = latency_std / avg_latency  # Coefficient of Variation
    throughput    = 1.0 / avg_latency
    cpu_usage     = psutil.cpu_percent(interval=0.1)
    memory_usage  = psutil.virtual_memory().percent
    sla_breach    = 1 if p95_latency > SLA_THRESHOLD_P95 else 0

    return {
        "run_id"        : str(uuid.uuid4())[:8],
        "wall_clock_time": datetime.utcnow().isoformat(),
        "batch_size"    : batch_size,
        "repeat_index"  : repeat_index,
        "avg_latency"   : round(avg_latency, 6),
        "p95_latency"   : round(p95_latency, 6),
        "p99_latency"   : round(p99_latency, 6),
        "latency_std"   : round(latency_std, 6),
        "cv_latency"    : round(cv_latency, 4),
        "throughput"    : round(throughput, 4),
        "cpu_usage"     : cpu_usage,
        "memory_usage"  : memory_usage,
        "sla_breach"    : sla_breach
    }

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
def generate_dataset():
    records = []
    total = len(BATCH_SIZES) * REPEATS_PER_BATCH
    count = 0

    print(f"Starting telemetry generation — {total} experiments\n")

    for batch_size in BATCH_SIZES:
        for repeat in range(REPEATS_PER_BATCH):
            record = run_experiment(batch_size, repeat)
            records.append(record)
            count += 1
            print(f"[{count}/{total}] batch={batch_size} | "
                  f"avg={record['avg_latency']:.4f}s | "
                  f"p95={record['p95_latency']:.4f}s | "
                  f"cpu={record['cpu_usage']}% | "
                  f"sla={'BREACH' if record['sla_breach'] else 'OK'}")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Dataset saved → {OUTPUT_FILE}")
    print(f"   Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"\nColumns: {list(df.columns)}")
    return df

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    df = generate_dataset()