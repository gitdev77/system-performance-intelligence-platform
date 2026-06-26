# day42_llm_alerts.py
# Day 42: LLM Layer — Gemini generates natural language alerts
import time
from google import genai
import pandas as pd
from datetime import datetime

# ── 1. Configure Gemini ───────────────────────────────────────────────────
API_KEY = "AQ.Ab8RN6K-0_jPyO3IOFAq-ylNQSRw0ANQOKhuV1o-dSbyRZFPBA"
client = genai.Client(api_key=API_KEY)
print("✅ Gemini connected")

# ── 2. Load run-level data (individual runs, not summary) ─────────────────
df = pd.read_csv("day35_runs.csv")
summary = pd.read_csv("day35_summary.csv")
print(f"✅ Loaded {len(df)} runs across {df['batch_size'].nunique()} batch sizes")

# ── 3. Define alert rules ─────────────────────────────────────────────────
# These mirror your Day 27-28 MLOps monitoring logic
SLA_THRESHOLD_MS  = 20.0
WARN_THRESHOLD    = 15.0   # yellow warning before breach
COST_ALERT_USD    = 0.01   # alert if cost > 1 cent per request
CPU_ALERT_PCT     = 80.0   # alert if CPU > 80%

# ── 4. Detect alert conditions from real data ─────────────────────────────
alerts = []

for _, row in summary.iterrows():
    batch = int(row['batch_size'])
    p95   = row['p95_latency_ms']
    avg   = row['avg_latency_ms']
    cpu   = row['avg_cpu_pct']
    cost  = row['avg_cost_usd']
    breach_rate = row['breach_rate_pct']
    regime = row['regime']

    # Determine severity
    if breach_rate >= 50:
        severity = "CRITICAL"
    elif breach_rate > 0:
        severity = "WARNING"
    elif p95 > WARN_THRESHOLD:
        severity = "CAUTION"
    else:
        severity = "OK"

    if severity != "OK":
        alerts.append({
            "batch_size"   : batch,
            "severity"     : severity,
            "p95_ms"       : round(p95, 2),
            "avg_ms"       : round(avg, 2),
            "breach_rate"  : breach_rate,
            "cpu_pct"      : round(cpu, 1),
            "cost_usd"     : round(cost, 6),
            "regime"       : regime,
        })

print(f"\n⚠️  {len(alerts)} alert conditions detected\n")

# ── 5. Generate natural language alert for each condition ─────────────────
alert_log = []

for alert in alerts:
    prompt = f"""
You are an ML infrastructure monitoring system generating a concise alert message.

Alert data:
- Batch size: {alert['batch_size']}
- Severity: {alert['severity']}
- P95 latency: {alert['p95_ms']}ms (SLA threshold: {SLA_THRESHOLD_MS}ms)
- Avg latency: {alert['avg_ms']}ms
- SLA breach rate: {alert['breach_rate']}%
- CPU usage: {alert['cpu_pct']}%
- Cost per request: ${alert['cost_usd']}
- Performance regime: {alert['regime']}

Generate a single alert message with:
1. Severity emoji (🔴 CRITICAL / 🟡 WARNING / 🟠 CAUTION)
2. One-line summary of the issue
3. Key metric that triggered the alert
4. One specific recommended action

Keep it under 80 words. Be direct and actionable.
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    alert_text = response.text.strip()
    time.sleep(3)

    alert_log.append({
        "timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batch_size": alert['batch_size'],
        "severity"  : alert['severity'],
        "message"   : alert_text
    })

    print(f"{'='*60}")
    print(f"[{alert['severity']}] Batch={alert['batch_size']}")
    print(alert_text)
    print()