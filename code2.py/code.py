import psutil
import time
import numpy as np

def heavy_computation(size):
    start = time.time()
    
    # Heavy matrix multiplication
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    C = np.dot(A, B)
    
    end = time.time()
    
    latency = end - start
    return latency

print("Starting workload experiment...\n")

batch_sizes = [200, 400, 600]  # Increase to increase load

for size in batch_sizes:
    print(f"\nRunning workload with matrix size: {size}")
    
    cpu_before = psutil.cpu_percent(interval=1)
    memory_before = psutil.virtual_memory().percent
    
    latency = heavy_computation(size)
    
    cpu_after = psutil.cpu_percent(interval=1)
    memory_after = psutil.virtual_memory().percent
    
    print(f"Latency: {latency:.4f} seconds")
    print(f"CPU Usage: {cpu_after}%")
    print(f"Memory Usage: {memory_after}%")
    print("-" * 40)


    