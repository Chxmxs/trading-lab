import pandas as pd
import numpy as np

from trading_lab.splits import build_split_masks, summarize_masks
from trading_lab import ensure_utc_index

# Build a dummy OHLCV frame at 1H freq across full dataset window
idx = pd.date_range("2017-01-01", "2025-03-14 23:59:59", freq="1H", tz="UTC")
df = pd.DataFrame({
    "open":   np.ones(len(idx)),
    "high":   np.ones(len(idx))*1.1,
    "low":    np.ones(len(idx))*0.9,
    "close":  np.ones(len(idx)),
    "volume": np.ones(len(idx))*10,
}, index=idx)

df = ensure_utc_index(df)

masks = build_split_masks(
    df,
    lookback_bars=72,          # e.g., 72 bars lookback at 1H
    label_horizon_bars=24,     # e.g., predict next 24 hours
    data_freq="1H"
)

print("Counts:", summarize_masks(masks))
for part in ("train","validation","test"):
    sel = df.index[masks[part]]
    if len(sel):
        print(part, "range:", sel.min(), "?", sel.max(), "n=", len(sel))
    else:
        print(part, "range: (empty)")