import pandas as pd

def strategy(df: pd.DataFrame, run_config: dict, lookback: int = 20, threshold: float = 1.0):
    price = df["close"].astype(float)
    rolling_mean = price.rolling(window=lookback, min_periods=1).mean()
    signals = (price < threshold * rolling_mean).astype(int)
    returns = price.pct_change().fillna(0.0)
    strat_returns = returns * signals.shift(1).fillna(0)
    equity = (1 + strat_returns).cumprod()
    signal_diff = signals.diff().fillna(0)
    trades = pd.DataFrame({"timestamp": df.index, "pnl_pct": strat_returns})
    trades = trades[signal_diff != 0].reset_index(drop=True)
    return {"equity": equity, "trades": trades}
