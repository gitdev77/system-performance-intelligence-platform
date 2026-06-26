# day41_llm_report.py
# Day 41: LLM Layer — Gemini generates natural language performance report

from google import genai
import pandas as pd
from datetime import datetime

# ── 1. Configure Gemini ───────────────────────────────────────────────────
API_KEY = "AQ.Ab8RN6K-0_jPyO3IOFAq-ylNQSRw0ANQOKhuV1o-dSbyRZFPBA"  # paste your actual key here
client = genai.Client(api_key=API_KEY)
print("✅ Gemini 1.5 Flash connected")

# ── 2. Load your telemetry summary ────────────────────────────────────────
df = pd.read_csv("day35_summary.csv")
print(f"✅ Loaded summary: {len(df)} batch sizes")

# ── 3. Build structured context for Gemini ────────────────────────────────
data_context = """
SYSTEM: ML Inference Workload — Matrix Multiply A(batch×512) × B(512×512)
PLATFORM: Apple Silicon Mac
SLA THRESHOLD: 20ms on P95 latency
MEASUREMENT: 8 runs per batch size, 13 batch sizes total

TELEMETRY SUMMARY:
"""

for _, row in df.iterrows():
    breach_flag = "🔴 BREACH ZONE" if row['breach_rate_pct'] > 0 else "✅ SAFE"
    data_context += f"""
Batch={int(row['batch_size'])}: avg_latency={row['avg_latency_ms']:.2f}ms | p95={row['p95_latency_ms']:.2f}ms | breach_rate={row['breach_rate_pct']:.1f}% | cpu={row['avg_cpu_pct']:.1f}% | throughput={row['avg_throughput']:.1f} ops/s | cost=${row['avg_cost_usd']:.6f} | regime={row['regime']} {breach_flag}"""

data_context += """

KEY FINDINGS FROM ANALYSIS:
- Throughput-latency correlation: r = -0.982 (throughput collapse is primary driver)
- CPU correlation with latency: r = -0.137 (CPU is NOT the bottleneck)
- M/M/1 queuing theory validation: r = 0.974
- Performance cliff detected at batch=192
- Cost cliff: batch=512 costs 6,314x more than batch=8
- Annual saving from optimal batch selection: $356,441 at 1M daily requests
- Recommended operating point: batch=8 or batch=16
"""

# ── 4. Build the prompt ───────────────────────────────────────────────────
prompt = f"""
You are a senior ML infrastructure engineer analyzing a performance intelligence report.

Here is the telemetry data from a real ML inference system:

{data_context}

Please write a comprehensive but concise performance analysis report with these sections:

1. EXECUTIVE SUMMARY (3-4 sentences for a non-technical manager)
2. PERFORMANCE CLIFF ANALYSIS (what happens at batch=192 and why)
3. COST IMPLICATIONS (explain the cost cliff in business terms)
4. RISK ASSESSMENT (which batch sizes are safe vs dangerous and why)
5. RECOMMENDATIONS (specific, actionable, prioritized)
6. CONCLUSION (one paragraph)

Write in a professional tone. Use specific numbers from the data.
Keep the total report under 600 words.
"""

# ── 5. Call Gemini API ────────────────────────────────────────────────────
print("\n🤖 Generating performance report with Gemini 2.5 Flash...")
print("   (this takes 5-10 seconds)\n")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)
report = response.text

# ── 6. Display and save ───────────────────────────────────────────────────
print("="*60)
print("SYSTEM PERFORMANCE INTELLIGENCE REPORT")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Model: Gemini 1.5 Flash")
print("="*60)
print(report)
print("="*60)

# Save to file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"day41_report_{timestamp}.txt"

with open(filename, "w") as f:
    f.write("SYSTEM PERFORMANCE INTELLIGENCE REPORT\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("Model: Gemini 1.5 Flash\n")
    f.write("="*60 + "\n\n")
    f.write(report)

print(f"\n✅ Report saved to: {filename}")
print(f"   Word count: ~{len(report.split()):,} words")