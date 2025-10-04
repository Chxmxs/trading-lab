from __future__ import annotations

import pandas as pd
from strategies.base import StrategyBase
from strategies.types import RunConfig

# tiny DF with UTC timestamps and one symbol
idx = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
df = pd.DataFrame({
    "date": idx,
    "symbol": "TEST",
    "open":   [100, 101, 102],
    "high":   [101, 102, 103],
    "low":    [ 99, 100, 101],
    "close":  [100.5, 101.5, 102.5],
    "volume": [1_000, 2_000, 1_500]
})

cfg = RunConfig(
    start=idx[0], end=idx[-1],
    initial_cash=100_000, fees=None, slippage=None,
    symbols=["TEST"], timeframe="1d"
)

base = StrategyBase("hello_world", cfg)
res = base.run(df)
print(res.equity.index.tz, res.equity.shape)
print(res.trades.columns.tolist(), len(res.trades))
