import psutil
import time
import numpy as np
import csv
from datetime import datetime


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

    return avg_latency, cpu_usage, memory_usage




    
def main():

    batch_sizes = [200, 400, 600, 800]

    with open("telemetry_week1.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "batch_size",
            "avg_latency",
            "cpu_usage",
            "memory_usage"
        ])

        import random

        for _ in range(30):
            size = random.choice(batch_sizes)

            avg_latency, cpu, memory = run_experiment(size)

            writer.writerow([
                datetime.now(),
                size,
                avg_latency,
                cpu,
                memory
            ])

            print(f"Batch: {size} | Latency: {avg_latency:.4f}s | CPU: {cpu}% | Memory: {memory}%")
            print("-" * 40)
if __name__ == "__main__":
    main()