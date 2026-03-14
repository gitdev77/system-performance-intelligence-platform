import psutil
import time
import torch
import torch.nn as nn
import csv
import random
from datetime import datetime

# ----------------------------
# Configuration
# ----------------------------
SLA_THRESHOLD = 0.05  # seconds (adjust if needed)
EXPERIMENT_RUNS = 80

batch_sizes = [64, 128, 256, 512, 1024]

# ----------------------------
# Simple Neural Network Model
# ----------------------------
model = nn.Sequential(
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)

model.eval()  # inference mode


def heavy_computation(batch_size):
    start = time.time()

    input_tensor = torch.randn(batch_size, 512)
    with torch.no_grad():
        output = model(input_tensor)

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

    throughput = 1 / avg_latency
    sla_breach = 1 if avg_latency > SLA_THRESHOLD else 0

    return avg_latency, cpu_usage, memory_usage, throughput, sla_breach


def main():

    with open("telemetry_week1.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "batch_size",
            "avg_latency",
            "throughput",
            "cpu_usage",
            "memory_usage",
            "sla_breach"
        ])

        for _ in range(EXPERIMENT_RUNS):

            size = random.choice(batch_sizes)

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