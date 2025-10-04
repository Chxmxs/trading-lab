class BreakoutStrategy:
    def __init__(self, **params):
        self.params = params

    def run(self, df, run_config=None):
        import pandas as pd
        if df is None:
            idx = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
        else:
            idx = df.index
        equity = pd.Series(100.0, index=idx, name="equity")
        trades = pd.DataFrame(columns=["timestamp", "pnl_pct"])
        return {"equity": equity, "trades": trades}
