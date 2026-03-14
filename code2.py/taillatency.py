import torch
import torch.nn as nn
import time
import psutil
import csv
from datetime import datetime
import random

# ================================
# 1️⃣ Configuration
# ================================

SLA_THRESHOLD = 0.05  # seconds (adjust later if needed)
REPETITIONS = 15      # number of runs per batch size

batch_sizes = [64, 128, 256, 512, 1024]

# ================================
# 2️⃣ Simple Neural Network Model
# ================================

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleModel()

# ================================
# 3️⃣ Inference Simulation
# ================================

def heavy_computation(batch_size):
    input_tensor = torch.randn(batch_size, 512)

    start = time.time()
    _ = model(input_tensor)
    end = time.time()

    return end - start


# ================================
# 4️⃣ Experiment Runner
# ================================

def run_experiment(batch_size):
    latencies = []

    for _ in range(REPETITIONS):
        latency = heavy_computation(batch_size)
        latencies.append(latency)

    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent

    return latencies, cpu_usage, memory_usage


# ================================
# 5️⃣ Main Execution Loop
# ================================

with open("telemetry_week1.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "timestamp",
        "batch_size",
        "avg_latency",
        "p95_latency",
        "p99_latency",
        "throughput",
        "cpu_usage",
        "memory_usage",
        "sla_breach"
    ])

    for _ in range(30):   # 30 experiments
        size = random.choice(batch_sizes)

        latencies, cpu, memory = run_experiment(size)

        latencies_sorted = sorted(latencies)

        avg_latency = sum(latencies) / len(latencies)

        p95_latency = latencies_sorted[int(0.95 * len(latencies))]
        p99_latency = latencies_sorted[int(0.99 * len(latencies))]

        throughput = 1 / avg_latency

        # 🔥 SLA based on P95 (not average)
        sla_breach = 1 if p95_latency > SLA_THRESHOLD else 0

        writer.writerow([
            datetime.now(),
            size,
            avg_latency,
            p95_latency,
            p99_latency,
            throughput,
            cpu,
            memory,
            sla_breach
        ])

print("Experiment Complete.")
