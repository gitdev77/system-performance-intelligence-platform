import psutil
import time
import numpy as np
import csv
from datetime import datetime
import random   # 🔵 NEW (for random batch selection)

# 🔵 NEW: SLA threshold definition
SLA_THRESHOLD = 0.15   # latency must be <= 0.15 sec


def heavy_computation(size):
    start = time.time()
    
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    C = np.dot(A, B)
    
    end = time.time()
    
    return end - start


def run_experiment(batch_size, repetitions=5):
    latencies = []

    for _ in range(repetitions):
        latency = heavy_computation(batch_size)
        latencies.append(latency)

    avg_latency = sum(latencies) / len(latencies)

    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent

    # 🔵 NEW: Throughput calculation
    throughput = 1 / avg_latency

    # 🔵 NEW: SLA breach logic
    sla_breach = 1 if avg_latency > SLA_THRESHOLD else 0

    # 🟡 MODIFIED: Return new values
    return avg_latency, cpu_usage, memory_usage, throughput, sla_breach


def main():

    batch_sizes = [200, 400, 600, 800]

    with open("telemetry_week1.csv", "w", newline="") as file:
        writer = csv.writer(file)

        # 🟡 MODIFIED: Updated header
        writer.writerow([
            "timestamp",
            "batch_size",
            "avg_latency",
            "throughput",
            "cpu_usage",
            "memory_usage",
            "sla_breach"
        ])

        # 🟡 MODIFIED: Randomized experiments (80 runs)
        for _ in range(80):
            size = random.choice(batch_sizes)

            # 🟡 MODIFIED: Unpack new return values
            avg_latency, cpu, memory, throughput, sla_breach = run_experiment(size)

            writer.writerow([
                datetime.now(),
                size,
                avg_latency,
                throughput,
                cpu,
                memory,
                sla_breach
            ])

            print(f"Batch: {size}")
            print(f"Latency: {avg_latency:.4f}s")
            print(f"Throughput: {throughput:.2f} req/sec")
            print(f"CPU: {cpu}% | Memory: {memory}%")
            print(f"SLA Breach: {sla_breach}")
            print("-" * 50)


if __name__ == "__main__":
    main()