# -*- coding: utf-8 -*-
"""scripts/labels_generate_mlfl.py — generate labels via mlfinlab (run in tradebot-mlfl)."""
import sys, json
from pathlib import Path
import pandas as pd

def main(symbol="BTCUSD", timeframe="15m", horizon=10, pt=2.0, sl=2.0):
    try:
        from mlfinlab.labeling import fixed_time_horizon
    except Exception as e:
        print("ERROR: mlfinlab not available in this env.", e)
        sys.exit(1)

    repo = Path(__file__).resolve().parents[1]
    price_path = repo / "data" / "usable" / f"ohlcv_{symbol}@{timeframe}.csv"
    if not price_path.exists():
        print("Missing:", price_path)
        sys.exit(2)

    df = pd.read_csv(price_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    labels = fixed_time_horizon(
        close=df["close"].values, threshold=0.0, window=horizon
    )
    out = pd.DataFrame({"timestamp": df["timestamp"].values[:len(labels)], "label": labels})
    out_path = repo / "data" / "parquet" / "labels" / f"{symbol}@{timeframe}__tb_labels.parquet"
    out.to_parquet(out_path, index=False)
    print("Wrote labels:", out_path)

if __name__ == "__main__":
    kw = {}
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k,v = arg.split("=",1)
            kw[k] = v
    main(**kw)
