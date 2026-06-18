# day34_sql.py

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import json

# ─────────────────────────────────────────
# DAY 34 — SQL LAYER
# PostgreSQL analytics on telemetry data
#
# GOAL: Load telemetry into PostgreSQL,
# create proper schema, run analytics
# queries that answer business questions
# using pure SQL.
# ─────────────────────────────────────────

DATA_FILE    = "/Users/devansh/Desktop/optimizeation/telemetry_v2.csv"
VERDICT_FILE = "/Users/devansh/Desktop/optimizeation/day20_verdicts.csv"
COST_FILE    = "/Users/devansh/Desktop/optimizeation/day30_cost_report.json"

# ── DB CONNECTION ──
DB_CONFIG = {
    "dbname"  : "system_perf_db",
    "user"    : "devansh",
    "host"    : "localhost",
    "port"    : 5432,
}

SLA_THRESHOLD_MS = 20.0
MATRIX_DIM       = 512
BYTES_PER_FLOAT  = 4
CPU_COST_PER_SEC = 0.05 / 3600
SLA_PENALTY      = 0.50

# ─────────────────────────────────────────
# STEP 1 — Load & prepare data
# ─────────────────────────────────────────
df = pd.read_csv(DATA_FILE)

df["cpu_usage_clean"] = df["cpu_usage"].replace(0.0, np.nan)
df["cpu_usage_clean"] = (
    df["cpu_usage_clean"]
    .rolling(window=5, min_periods=1, center=True)
    .median().ffill().bfill()
)
df["avg_latency_ms"] = df["avg_latency"] * 1000
df["p95_latency_ms"] = df["p95_latency"] * 1000
df["sla_breach_new"] = (
    df["avg_latency_ms"] > SLA_THRESHOLD_MS
).astype(int)

throughput_peak = df.groupby("batch_size")["throughput"].mean().max()
df["rho"] = (1 - df["throughput"] / throughput_peak).clip(lower=0)

b           = df["batch_size"]
flops       = 2 * b * MATRIX_DIM * MATRIX_DIM
bytes_moved = ((b * MATRIX_DIM) + (MATRIX_DIM * MATRIX_DIM) +
               (b * MATRIX_DIM)) * BYTES_PER_FLOAT
df["AI"] = flops / bytes_moved

print(f"✅ Step 1 — Data prepared ({len(df)} rows)\n")

# ─────────────────────────────────────────
# STEP 2 — Load verdicts (Day 20) and
# cost report (Day 30) for batch_summary
# ─────────────────────────────────────────
verdicts_df = pd.read_csv(VERDICT_FILE)

with open(COST_FILE) as f:
    cost_data = json.load(f)

cost_df = pd.DataFrame(cost_data["per_batch_costs"])

print(f"✅ Step 2 — Verdicts and cost data loaded\n")

# ─────────────────────────────────────────
# STEP 3 — Connect to PostgreSQL
# ─────────────────────────────────────────
conn = psycopg2.connect(**DB_CONFIG)
cur  = conn.cursor()

print(f"✅ Step 3 — Connected to PostgreSQL "
      f"(system_perf_db)\n")

# ─────────────────────────────────────────
# STEP 4 — CREATE TABLES
#
# DROP IF EXISTS = clean slate every run
# (safe for development — in production
#  you'd use migrations instead)
# ─────────────────────────────────────────
cur.execute("DROP TABLE IF EXISTS batch_summary CASCADE;")
cur.execute("DROP TABLE IF EXISTS telemetry_runs CASCADE;")

# Table 1: every individual run
cur.execute("""
    CREATE TABLE telemetry_runs (
        run_id          SERIAL PRIMARY KEY,
        batch_size      INTEGER NOT NULL,
        cpu_usage       FLOAT,
        throughput      FLOAT,
        avg_latency_ms  FLOAT,
        p95_latency_ms  FLOAT,
        cv_latency      FLOAT,
        sla_breach      INTEGER,
        rho             FLOAT,
        AI              FLOAT,
        created_at      TIMESTAMP DEFAULT NOW()
    );
""")

# Table 2: aggregated per batch size
cur.execute("""
    CREATE TABLE batch_summary (
        batch_size          INTEGER PRIMARY KEY,
        avg_latency_ms      FLOAT,
        p95_latency_ms      FLOAT,
        avg_throughput      FLOAT,
        sla_breach_pct      FLOAT,
        bottleneck_type     VARCHAR(50),
        severity            VARCHAR(20),
        cost_per_request    FLOAT,
        total_cost          FLOAT,
        efficiency_score    FLOAT,
        rho_avg             FLOAT,
        AI_value            FLOAT
    );
""")

conn.commit()
print(f"✅ Step 4 — Tables created\n")

# ─────────────────────────────────────────
# STEP 5 — INSERT DATA
# ─────────────────────────────────────────

# Insert telemetry_runs (104 rows)
runs_data = [
    (
        int(row["batch_size"]),
        float(row["cpu_usage_clean"]),
        float(row["throughput"]),
        float(row["avg_latency_ms"]),
        float(row["p95_latency_ms"]),
        float(row["cv_latency"]),
        int(row["sla_breach_new"]),
        float(row["rho"]),
        float(row["AI"]),
    )
    for _, row in df.iterrows()
]

execute_values(cur, """
    INSERT INTO telemetry_runs
    (batch_size, cpu_usage, throughput, avg_latency_ms,
     p95_latency_ms, cv_latency, sla_breach, rho, AI)
    VALUES %s
""", runs_data)

# Build batch summary rows
agg = (
    df.groupby("batch_size")
    .agg(
        avg_latency_ms = ("avg_latency_ms", "mean"),
        p95_latency_ms = ("p95_latency_ms", "mean"),
        avg_throughput = ("throughput",      "mean"),
        sla_breach_pct = ("sla_breach_new",  "mean"),
        rho_avg        = ("rho",             "mean"),
    )
    .reset_index()
)
agg["sla_breach_pct"] = agg["sla_breach_pct"] * 100

# Merge with verdicts
agg = agg.merge(
    verdicts_df[["batch_size", "bottleneck_type", "severity"]],
    on="batch_size", how="left"
)

# Merge with cost
agg = agg.merge(
    cost_df[["batch_size", "cost_per_request_usd",
             "total_cost_per_request"]],
    on="batch_size", how="left"
)

# AI value (deterministic from batch_size)
agg["AI_value"] = (
    2 * agg["batch_size"] * MATRIX_DIM * MATRIX_DIM /
    ((2 * agg["batch_size"] * MATRIX_DIM +
      MATRIX_DIM * MATRIX_DIM) * BYTES_PER_FLOAT)
)

# Efficiency score
throughput_peak_val = agg["avg_throughput"].max()
agg["efficiency_score"] = (
    agg["avg_throughput"] / throughput_peak_val
) * 100

summary_data = [
    (
        int(row["batch_size"]),
        float(row["avg_latency_ms"]),
        float(row["p95_latency_ms"]),
        float(row["avg_throughput"]),
        float(row["sla_breach_pct"]),
        str(row["bottleneck_type"]),
        str(row["severity"]),
        float(row["cost_per_request_usd"])
            if pd.notna(row["cost_per_request_usd"]) else 0.0,
        float(row["total_cost_per_request"])
            if pd.notna(row["total_cost_per_request"]) else 0.0,
        float(row["efficiency_score"]),
        float(row["rho_avg"]),
        float(row["AI_value"]),
    )
    for _, row in agg.iterrows()
]

execute_values(cur, """
    INSERT INTO batch_summary
    (batch_size, avg_latency_ms, p95_latency_ms,
     avg_throughput, sla_breach_pct, bottleneck_type,
     severity, cost_per_request, total_cost,
     efficiency_score, rho_avg, AI_value)
    VALUES %s
""", summary_data)

conn.commit()
print(f"✅ Step 5 — Data inserted")
print(f"   telemetry_runs : {len(runs_data)} rows")
print(f"   batch_summary  : {len(summary_data)} rows\n")

# ─────────────────────────────────────────
# STEP 6 — ANALYTICS QUERIES
# ─────────────────────────────────────────
print("=" * 65)
print("  DAY 34 — SQL ANALYTICS QUERIES")
print("=" * 65)

# ── Query 1: SLA breach summary ──
print(f"\n── Q1: Which batch sizes breach SLA? ──\n")
cur.execute("""
    SELECT
        batch_size,
        ROUND(sla_breach_pct::numeric, 1) AS breach_pct,
        severity,
        bottleneck_type
    FROM batch_summary
    WHERE sla_breach_pct > 0
    ORDER BY sla_breach_pct DESC;
""")
rows = cur.fetchall()
print(f"  {'Batch':<8} {'Breach%':<10} {'Severity':<12} {'Type'}")
print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*20}")
for r in rows:
    print(f"  {r[0]:<8} {r[1]:<10} {r[2]:<12} {r[3]}")

# ── Query 2: Most expensive batch sizes ──
print(f"\n── Q2: Top 3 most expensive batch sizes ──\n")
cur.execute("""
    SELECT
        batch_size,
        ROUND((total_cost * 1e6)::numeric, 3) AS total_cost_micro,
        ROUND(sla_breach_pct::numeric, 1)     AS breach_pct,
        severity
    FROM batch_summary
    ORDER BY total_cost DESC
    LIMIT 3;
""")
rows = cur.fetchall()
print(f"  {'Batch':<8} {'Cost(μ$)':<12} {'Breach%':<10} {'Severity'}")
print(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*10}")
for r in rows:
    print(f"  {r[0]:<8} {r[1]:<12} {r[2]:<10} {r[3]}")

# ── Query 3: Avg latency by bottleneck type ──
print(f"\n── Q3: Avg latency by bottleneck type ──\n")
cur.execute("""
    SELECT
        bottleneck_type,
        COUNT(*)                                   AS count,
        ROUND(AVG(avg_latency_ms)::numeric, 2)    AS avg_latency,
        ROUND(AVG(sla_breach_pct)::numeric, 1)    AS avg_breach_pct
    FROM batch_summary
    GROUP BY bottleneck_type
    ORDER BY avg_latency DESC;
""")
rows = cur.fetchall()
print(f"  {'Type':<25} {'Count':<8} {'Avg Latency':<14} {'Avg Breach%'}")
print(f"  {'-'*25} {'-'*8} {'-'*14} {'-'*12}")
for r in rows:
    print(f"  {r[0]:<25} {r[1]:<8} {r[2]:<14} {r[3]}")

# ── Query 4: Safe operating zone ──
print(f"\n── Q4: Safe operating zone ──\n")
cur.execute("""
    SELECT
        batch_size,
        ROUND(avg_latency_ms::numeric, 2)  AS latency_ms,
        ROUND(avg_throughput::numeric, 2)  AS throughput,
        ROUND(efficiency_score::numeric, 1) AS efficiency_pct
    FROM batch_summary
    WHERE sla_breach_pct = 0
      AND avg_latency_ms < 15
    ORDER BY efficiency_score DESC;
""")
rows = cur.fetchall()
print(f"  {'Batch':<8} {'Latency':<12} {'Throughput':<14} {'Efficiency%'}")
print(f"  {'-'*8} {'-'*12} {'-'*14} {'-'*12}")
for r in rows:
    print(f"  {r[0]:<8} {r[1]:<12} {r[2]:<14} {r[3]}")

# ── Query 5: Throughput efficiency ranking ──
print(f"\n── Q5: Throughput efficiency ranking ──\n")
cur.execute("""
    SELECT
        batch_size,
        ROUND(efficiency_score::numeric, 1)  AS efficiency_pct,
        ROUND(avg_latency_ms::numeric, 2)    AS latency_ms,
        ROUND(rho_avg::numeric, 3)           AS rho
    FROM batch_summary
    ORDER BY efficiency_score DESC
    LIMIT 5;
""")
rows = cur.fetchall()
print(f"  {'Batch':<8} {'Efficiency%':<14} {'Latency':<12} {'ρ'}")
print(f"  {'-'*8} {'-'*14} {'-'*12} {'-'*8}")
for r in rows:
    print(f"  {r[0]:<8} {r[1]:<14} {r[2]:<12} {r[3]}")

# ── Query 6: Individual runs with breach ──
print(f"\n── Q6: Individual breach runs from telemetry ──\n")
cur.execute("""
    SELECT
        run_id,
        batch_size,
        ROUND(avg_latency_ms::numeric, 2)  AS latency_ms,
        ROUND(cpu_usage::numeric, 1)       AS cpu,
        ROUND(throughput::numeric, 2)      AS throughput,
        ROUND(cv_latency::numeric, 3)      AS cv_latency
    FROM telemetry_runs
    WHERE sla_breach = 1
    ORDER BY avg_latency_ms DESC;
""")
rows = cur.fetchall()
print(f"  {'RunID':<8} {'Batch':<8} {'Latency':<12} "
      f"{'CPU%':<8} {'Throughput':<12} {'CV'}")
print(f"  {'-'*8} {'-'*8} {'-'*12} {'-'*8} {'-'*12} {'-'*8}")
for r in rows:
    print(f"  {r[0]:<8} {r[1]:<8} {r[2]:<12} "
          f"{r[3]:<8} {r[4]:<12} {r[5]}")

# ── Query 7: SLA breach rate by batch group ──
print(f"\n── Q7: SLA breach rate by batch group ──\n")
cur.execute("""
    SELECT
        CASE
            WHEN batch_size <= 64  THEN 'Small  (8-64)'
            WHEN batch_size <= 192 THEN 'Medium (96-192)'
            ELSE                        'Large  (256-512)'
        END                                        AS batch_group,
        COUNT(*)                                   AS runs,
        SUM(sla_breach)                            AS breaches,
        ROUND(
            AVG(sla_breach) * 100, 1
        )                                          AS breach_rate_pct,
        ROUND(AVG(avg_latency_ms)::numeric, 2)    AS avg_latency
    FROM telemetry_runs
    GROUP BY batch_group
    ORDER BY breach_rate_pct DESC;
""")
rows = cur.fetchall()
print(f"  {'Group':<20} {'Runs':<8} {'Breaches':<10} "
      f"{'Breach%':<10} {'Avg Latency'}")
print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")
for r in rows:
    print(f"  {r[0]:<20} {r[1]:<8} {r[2]:<10} "
          f"{r[3]:<10} {r[4]}")

# ── Query 8: batch=192 anomaly deep-dive ──
print(f"\n── Q8: batch=192 anomaly deep-dive ──\n")
cur.execute("""
    SELECT
        batch_size,
        ROUND(avg_latency_ms::numeric, 2)  AS latency_ms,
        ROUND(cpu_usage::numeric, 1)       AS cpu,
        ROUND(throughput::numeric, 2)      AS throughput,
        ROUND(cv_latency::numeric, 3)      AS cv,
        sla_breach
    FROM telemetry_runs
    WHERE batch_size = 192
    ORDER BY avg_latency_ms DESC;
""")
rows = cur.fetchall()
print(f"  {'Batch':<8} {'Latency':<12} {'CPU%':<8} "
      f"{'Throughput':<12} {'CV':<8} {'Breach'}")
print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*12} {'-'*8} {'-'*8}")
for r in rows:
    breach_flag = "⚠️ YES" if r[5] == 1 else "OK"
    print(f"  {r[0]:<8} {r[1]:<12} {r[2]:<8} "
          f"{r[3]:<12} {r[4]:<8} {breach_flag}")

# ─────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────
cur.close()
conn.close()

print(f"\n{'='*65}")
print(f"  Day 34 complete.")
print(f"  Database: system_perf_db")
print(f"  Tables  : telemetry_runs (104 rows), "
      f"batch_summary (13 rows)")
print(f"  Queries : 8 analytics queries executed")
print(f"{'='*65}\n")
