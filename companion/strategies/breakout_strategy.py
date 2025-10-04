import pandas as pd

def strategy(df: pd.DataFrame, run_config: dict, lookback: int = 20, breakout_factor: float = 1.0):
    """
    Enter long when the close exceeds the previous rolling high multiplied by breakout_factor.
    Exit (flat) when the signal drops to zero. This ignores short positions.
    Parameters:
    - df: DataFrame with columns 'open', 'high', 'low', 'close', 'volume'
    - run_config: unused here but accepted for compatibility
    - lookback: window size for computing the rolling high
    - breakout_factor: multiplier to adjust breakout threshold (e.g. 1.0 for a strict breakout)
    """
    # Ensure required columns exist
    for col in ('open', 'high', 'low', 'close'):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from input DataFrame.")
    close = df['close'].astype(float)
    high = df['high'].astype(float)

    # Compute rolling maximum of highs
    rolling_high = high.rolling(window=lookback, min_periods=1).max()

    # Generate long signals when close > breakout_factor * previous rolling high
    # Shift rolling_high by 1 to avoid look-ahead bias
    threshold = breakout_factor * rolling_high.shift(1).fillna(rolling_high)
    signals = (close > threshold).astype(int)

    # Calculate returns and apply strategy signals
    returns = close.pct_change().fillna(0.0)
    strat_returns = returns * signals.shift(1).fillna(0)

    # Equity curve (starting at 1.0)
    equity = (1 + strat_returns).cumprod()

    # Identify trade entries/exits
    signal_diff = signals.diff().fillna(0)
    trades = pd.DataFrame({'timestamp': df.index, 'pnl_pct': strat_returns})
    trades = trades[signal_diff != 0].reset_index(drop=True)

    return {'equity': equity, 'trades': trades}
