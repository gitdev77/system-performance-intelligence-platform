import psutil
import time

print("Starting system monitoring...\n")

for i in range(10):
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    
    print(f"Iteration {i+1}")
    print(f"CPU Usage: {cpu}%")
    print(f"Memory Usage: {memory}%")
    print("-" * 30)
    
    time.sleep(1)
