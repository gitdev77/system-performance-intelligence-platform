# day43_llm_recommend.py
# Day 43: LLM Layer — Gemini auto recommendation engine

from google import genai
import pandas as pd
import json
import time
from datetime import datetime

# ── 1. Configure Gemini ───────────────────────────────────────────────────
API_KEY = "AQ.Ab8RN6K-0_jPyO3IOFAq-ylNQSRw0ANQOKhuV1o-dSbyRZFPBA"
client = genai.Client(api_key=API_KEY)
print("✅ Gemini connected")

# ── 2. Load all data sources ──────────────────────────────────────────────
summary = pd.read_csv("day35_summary.csv")
runs    = pd.read_csv("day35_runs.csv")
print(f"✅ Loaded {len(summary)} batch sizes, {len(runs)} total runs")

# ── 3. Load Day 31 recommendation (your Pareto analysis) ─────────────────
try:
    with open("day31_recommendation.json") as f:
        pareto = json.load(f)
    pareto_str = json.dumps(pareto, indent=2)
except:
    pareto_str = "Pareto analysis: batch=8 recommended across all 3 business scenarios"

# ── 4. Build full system context ──────────────────────────────────────────
context = f"""
SYSTEM CONTEXT:
- Workload: ML inference, matrix multiply A(batch×512) × B(512×512)
- Platform: Apple Silicon Mac
- SLA threshold: 20ms P95 latency
- Dataset: 104 runs, 13 batch sizes [8,16,32,48,64,96,128,192,256,320,384,448,512]

PERFORMANCE SUMMARY BY BATCH SIZE:
{summary[['batch_size','avg_latency_ms','p95_latency_ms','breach_rate_pct','avg_cpu_pct','avg_throughput','avg_cost_usd','regime']].to_string(index=False)}

STATISTICAL FINDINGS:
- Throughput-latency correlation: r = -0.982
- CPU-latency correlation: r = -0.137 (CPU NOT bottleneck)
- M/M/1 queuing theory fit: r = 0.974
- Performance cliff: batch=192 (50% breach rate)
- Cost cliff: batch=512 = 6,314x more expensive than batch=8
- Annual saving at 1M daily requests: $356,441

PARETO ANALYSIS RESULT:
{pareto_str}

CURRENT ALERT STATUS:
- 2 CRITICAL alerts (batch=192, batch=512)
- 6 WARNING alerts (batch=96,128,256,320,384,448)
- Safe zone: batch <= 64
"""

# ── 5. Three recommendation scenarios ────────────────────────────────────
scenarios = [
    {
        "name": "Scenario A — Maximum Performance",
        "constraint": "Minimize latency at all costs. SLA compliance is non-negotiable.",
        "business": "Real-time user-facing API where every millisecond matters"
    },
    {
        "name": "Scenario B — Balanced Cost-Performance",
        "constraint": "Balance cost and performance. Allow up to 5% breach rate.",
        "business": "Internal ML pipeline with moderate latency tolerance"
    },
    {
        "name": "Scenario C — Maximum Throughput",
        "constraint": "Maximize operations per second within SLA limits.",
        "business": "Batch processing job that needs to process maximum requests per hour"
    }
]

recommendations = []

for scenario in scenarios:
    print(f"\n🤖 Generating recommendation for {scenario['name']}...")

    prompt = f"""
You are a senior ML infrastructure engineer making batch size recommendations.

{context}

BUSINESS SCENARIO:
- Name: {scenario['name']}
- Constraint: {scenario['constraint']}
- Use case: {scenario['business']}

Based on all the data above, provide a structured recommendation with exactly these sections:

RECOMMENDED BATCH SIZE: [single number]
CONFIDENCE: [High/Medium/Low]
REASONING: [2-3 sentences explaining why]
EXPECTED PERFORMANCE: [latency, breach rate, throughput at recommended batch]
EXPECTED COST: [cost per request and annual cost at 1M daily requests]
RISK: [what could go wrong]
ALTERNATIVE: [second-best option if primary isn't feasible]

Be specific with numbers. Use data from the telemetry summary above.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    rec_text = response.text.strip()
    recommendations.append({
        "scenario"      : scenario['name'],
        "constraint"    : scenario['constraint'],
        "recommendation": rec_text,
        "timestamp"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    print(f"\n{'='*60}")
    print(f"{scenario['name'].upper()}")
    print(f"{'='*60}")
    print(rec_text)

    time.sleep(4)  # avoid rate limiting

# ── 6. Final synthesis ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SYNTHESIZING FINAL RECOMMENDATION...")
print(f"{'='*60}\n")

synthesis_prompt = f"""
You are a senior ML infrastructure engineer writing an executive summary.

You have analyzed an ML inference system and produced 3 scenario-based recommendations.
All 3 scenarios recommend batch sizes in the range of 8-16.

Key facts:
- Batch=8: avg_latency=7.15ms, p95=9.61ms, breach_rate=0%, cost=$0.000008, throughput=140.5 ops/s
- Batch=16: avg_latency=7.20ms, p95=9.66ms, breach_rate=0%, cost=$0.000016, throughput=122.1 ops/s
- Batch=192: avg_latency=18.50ms, p95=29.38ms, breach_rate=50%, cost=$0.025192
- Batch=512: avg_latency=20.85ms, p95=28.61ms, breach_rate=100%, cost=$0.050512
- Annual saving from optimal selection: $356,441

Write a 150-word executive summary that:
1. States the single best recommended batch size
2. Explains the business case in non-technical terms
3. Quantifies the risk of NOT following the recommendation
4. Ends with one clear call to action

Write for a CTO or VP Engineering audience.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=synthesis_prompt
)
synthesis = response.text.strip()

print(synthesis)

# ── 7. Save everything ────────────────────────────────────────────────────
output = {
    "generated_at"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model"           : "gemini-2.5-flash",
    "scenarios"       : recommendations,
    "executive_summary": synthesis
}

with open("day43_recommendations.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print("✅ All recommendations saved to day43_recommendations.json")
print(f"   Scenarios analyzed: {len(recommendations)}")