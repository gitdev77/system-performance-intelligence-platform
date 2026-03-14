import pandas as pd

df = pd.read_csv("telemetry_week1.csv")

print("Total Rows:", len(df))
print("Total SLA Breaches:", df["sla_breach"].sum())
print("Overall Breach Rate (%):", round(df["sla_breach"].mean() * 100, 2))

print("\nBreach Rate by Batch Size:")
print((df.groupby("batch_size")["sla_breach"].mean() * 100).round(2))

print("\nAverage Latency by Batch Size:")
print(df.groupby("batch_size")["avg_latency"].mean().round(4))