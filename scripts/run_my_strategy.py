from __future__ import annotations

import pandas as pd

from strategies.my_strategy import MyStrategy
from strategies.types import RunConfig

def main() -> None:
    # Minimal dummy OHLCV (UTC)
    idx = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    df = pd.DataFrame({
        "date": idx,
        "symbol": "TEST",
        "open":   [100, 101, 102],
        "high":   [101, 102, 103],
        "low":    [ 99, 100, 101],
        "close":  [100.5, 101.5, 102.5],
        "volume": [1000, 2000, 1500],
    })

    cfg = RunConfig(
        start=idx[0],
        end=idx[-1],
        initial_cash=100_000.0,
        fees=None,
        slippage=None,
        symbols=["TEST"],
        timeframe="1d",
    )

    strat = MyStrategy("my_strategy", cfg)
    res = strat.run(df)

    print("equity len:", len(res.equity), "tz:", res.equity.index.tz)
    print("trades cols:", list(res.trades.columns))
    print("trades rows:", len(res.trades))

if __name__ == "__main__":
    main()
