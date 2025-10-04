import os, pandas as pd
os.makedirs("data/metrics", exist_ok=True)

df = pd.read_csv("data/cleaned/ohlcv_BTCUSD@15m.csv")
ts = pd.to_datetime(df["date"], utc=True, errors="coerce").dropna()
if ts.empty:
    raise SystemExit("Could not parse timestamps from ohlcv_BTCUSD@15m.csv")

didx = pd.date_range(ts.min().normalize(), ts.max().normalize(), freq="D", tz="UTC")
out = pd.DataFrame({"date": didx, "sopr": 1.0})
out.to_csv("data/metrics/metric_btc.adjusted_sopr_asopr@1d.csv", index=False)
print("Wrote placeholder: data/metrics/metric_btc.adjusted_sopr_asopr@1d.csv  (sopr==1.0)")
