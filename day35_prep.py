# day35_prep.py  (v3 — correct units + correct SLA threshold)
import pandas as pd
import numpy as np

# ── 1. Load raw telemetry ──────────────────────────────────────────────────
df = pd.read_csv("telemetry_v2.csv")
print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

# ── 2. Convert latency from SECONDS → MILLISECONDS ─────────────────────────
# Raw CSV stores latency in seconds (e.g. 0.0184 = 18.4ms)
df["avg_latency_ms"] = df["avg_latency"] * 1000
df["p95_latency_ms"] = df["p95_latency"] * 1000
df["p99_latency_ms"] = df["p99_latency"] * 1000

# ── 3. Recompute SLA breach at correct threshold (20ms on p95) ────────────
# Original sla_breach column used 150ms — wrong for our analysis
# Day 25 recomputed at 20ms on p95, finding 7/104 breaches
SLA_THRESHOLD_MS = 20.0
df["sla_breach_v2"] = (df["p95_latency_ms"] > SLA_THRESHOLD_MS).astype(int)
df["sla_label"]     = df["sla_breach_v2"].map({0: "OK", 1: "BREACH"})

# ── 4. Cost model (from Day 30) ────────────────────────────────────────────
COMPUTE_RATE = 1e-6   # $ per item processed
SLA_PENALTY  = 0.05   # $ per breaching request

df["compute_cost_usd"]     = df["batch_size"] * COMPUTE_RATE
df["sla_penalty_usd"]      = df["sla_breach_v2"] * SLA_PENALTY
df["cost_per_request_usd"] = df["compute_cost_usd"] + df["sla_penalty_usd"]

# ── 5. Regime labels ───────────────────────────────────────────────────────
def assign_regime(b):
    if b <= 96:
        return "Safe"
    elif b <= 128:
        return "Warning"
    else:
        return "Danger"

df["regime"] = df["batch_size"].apply(assign_regime)

# ── 6. Run number within each batch size (1–8) ─────────────────────────────
df["run_number"] = df.groupby("batch_size").cumcount() + 1

# ── 7. Summary stats per batch size ───────────────────────────────────────
summary = (
    df.groupby("batch_size")
    .agg(
        avg_latency_ms  = ("avg_latency_ms",       "mean"),
        p95_latency_ms  = ("p95_latency_ms",        "mean"),
        p99_latency_ms  = ("p99_latency_ms",        "mean"),
        avg_throughput  = ("throughput",            "mean"),
        avg_cpu_pct     = ("cpu_usage",             "mean"),
        breach_rate_pct = ("sla_breach_v2",         lambda x: x.mean() * 100),
        avg_cost_usd    = ("cost_per_request_usd",  "mean"),
    )
    .reset_index()
)
summary["regime"] = summary["batch_size"].apply(assign_regime)

# ── 8. Save both files ─────────────────────────────────────────────────────
df.to_csv("day35_runs.csv", index=False)
summary.to_csv("day35_summary.csv", index=False)

print("\n✅ Files saved:")
print("   day35_runs.csv    — 104 rows, all individual runs (for heatmap)")
print("   day35_summary.csv — 13 rows, one per batch size (for trend charts)")

print("\n── Summary preview ───────────────────────────────────────────────────")
print(summary[["batch_size","avg_latency_ms","p95_latency_ms",
               "breach_rate_pct","avg_cost_usd","regime"]].to_string(index=False))

print("\n── Breach counts by regime ───────────────────────────────────────────")
print(df.groupby("regime")["sla_breach_v2"].agg(["sum","count"])
        .rename(columns={"sum":"breaches","count":"total_runs"}))

print("\n── Individual breach runs ────────────────────────────────────────────")
breaches = df[df["sla_breach_v2"] == 1][["batch_size","run_number","avg_latency_ms","p95_latency_ms"]]
print(breaches.to_string(index=False))