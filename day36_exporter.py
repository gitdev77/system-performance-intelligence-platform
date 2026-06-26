import time
import random
from prometheus_client import start_http_server, Gauge, Counter, Histogram

BATCH_SIZE = Gauge("inference_batch_size", "Current batch size")
LATENCY_MS = Gauge("inference_latency_ms", "Average latency ms")
P95_LATENCY_MS = Gauge("inference_p95_latency_ms", "P95 latency ms")
THROUGHPUT = Gauge("inference_throughput_ops", "Throughput ops/sec")
CPU_USAGE = Gauge("system_cpu_usage_pct", "CPU usage percent")
SLA_BREACH = Gauge("inference_sla_breach", "SLA breach flag")
REQUEST_COUNTER = Counter("inference_requests_total", "Total requests")
BREACH_COUNTER = Counter("inference_sla_breaches_total", "Total breaches")
LATENCY_HISTOGRAM = Histogram("inference_latency_histogram_ms", "Latency histogram", buckets=[5,8,10,12,15,18,20,25,30,50])

BATCH_SIZES = [8,16,32,48,64,96,128,192,256,320,384,448,512]
TELEMETRY = {
    8:   {"avg_lat":7.15,  "p95_lat":9.61,  "cpu":23.1, "throughput":134.0},
    16:  {"avg_lat":7.20,  "p95_lat":9.66,  "cpu":46.7, "throughput":122.1},
    32:  {"avg_lat":7.95,  "p95_lat":11.32, "cpu":56.9, "throughput":134.7},
    48:  {"avg_lat":9.15,  "p95_lat":12.59, "cpu":22.5, "throughput":122.4},
    64:  {"avg_lat":9.02,  "p95_lat":12.77, "cpu":57.8, "throughput":115.3},
    96:  {"avg_lat":10.16, "p95_lat":15.50, "cpu":79.9, "throughput":111.7},
    128: {"avg_lat":11.87, "p95_lat":21.36, "cpu":34.5, "throughput":98.0},
    192: {"avg_lat":18.50, "p95_lat":29.38, "cpu":57.5, "throughput":85.8},
    256: {"avg_lat":12.75, "p95_lat":17.61, "cpu":69.0, "throughput":76.9},
    320: {"avg_lat":15.21, "p95_lat":27.17, "cpu":23.6, "throughput":77.0},
    384: {"avg_lat":14.94, "p95_lat":20.56, "cpu":57.8, "throughput":66.5},
    448: {"avg_lat":16.00, "p95_lat":20.10, "cpu":91.4, "throughput":60.7},
    512: {"avg_lat":20.85, "p95_lat":28.61, "cpu":45.8, "throughput":54.5},
}
SLA_THRESHOLD_MS = 20.0

def simulate_inference(batch_size):
    base = TELEMETRY[batch_size]
    noise = lambda x: x * (1 + random.uniform(-0.15, 0.15))
    avg_lat = noise(base["avg_lat"])
    p95_lat = noise(base["p95_lat"])
    cpu = min(100, noise(base["cpu"]))
    tput = noise(base["throughput"])
    is_breach = 1 if p95_lat > SLA_THRESHOLD_MS else 0
    return avg_lat, p95_lat, cpu, tput, is_breach

def run():
    start_http_server(8001)
    print("✅ Metrics server started at http://localhost:8001/metrics")
    print("   Simulating inference... (Ctrl+C to stop)")
    batch_index = 0
    while True:
        batch = BATCH_SIZES[batch_index % len(BATCH_SIZES)]
        batch_index += 1
        avg_lat, p95_lat, cpu, tput, is_breach = simulate_inference(batch)
        BATCH_SIZE.set(batch)
        LATENCY_MS.set(round(avg_lat, 3))
        P95_LATENCY_MS.set(round(p95_lat, 3))
        THROUGHPUT.set(round(tput, 3))
        CPU_USAGE.set(round(cpu, 3))
        SLA_BREACH.set(is_breach)
        REQUEST_COUNTER.inc()
        LATENCY_HISTOGRAM.observe(avg_lat)
        if is_breach:
            BREACH_COUNTER.inc()
        status = "🔴 BREACH" if is_breach else "🟢 OK"
        print(f"batch={batch:>3} | lat={avg_lat:>6.2f}ms | p95={p95_lat:>6.2f}ms | cpu={cpu:>5.1f}% | {status}")
        time.sleep(10)

if __name__ == "__main__":
    run()
