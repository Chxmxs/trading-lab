import os, numpy as np, pandas as pd
os.makedirs("data/metrics", exist_ok=True)

df = pd.read_csv("data/cleaned/ohlcv_BTCUSD@15m.csv")
ts = pd.to_datetime(df["date"], utc=True, errors="coerce").dropna()
didx = pd.date_range(ts.min().normalize(), ts.max().normalize(), freq="D", tz="UTC")

# Wobbly SOPR around 1.0: ~ +- 0.02 with slow sinusoid + tiny noise
t = np.arange(len(didx))
sopr = 1.0 + 0.02*np.sin(2*np.pi*t/90.0) + np.random.normal(0, 0.003, len(didx))
out = pd.DataFrame({"date": didx, "sopr": sopr})
out.to_csv("data/metrics/metric_btc.adjusted_sopr_asopr@1d.csv", index=False)
print("Wrote wobbly SOPR placeholder at data/metrics/metric_btc.adjusted_sopr_asopr@1d.csv")
