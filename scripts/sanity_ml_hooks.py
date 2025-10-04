import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from strategies.base import StrategyBase
from strategies.types import TRADE_COLUMNS

class DummyMLStrategy(StrategyBase):
    def propose_entries(self, df: pd.DataFrame):
        # One candidate at the middle timestamp
        idx = df.index
        mid = idx[len(idx)//2]
        s = pd.Series(False, index=idx)
        s.loc[mid] = True
        return s

    def filter_entries(self, entries, meta_probs=None):
        # Drop everything (simulate meta-label reject)
        if entries is None:
            return None
        if hasattr(entries, "copy"):
            df = entries.copy()
            df["is_candidate"] = False
            return df[df["is_candidate"]]
        return None

def build_dummy_df():
    # 3-row dummy DF with UTC index
    ts0 = pd.Timestamp("2025-01-01T00:00:00Z")
    idx = pd.date_range(ts0, periods=3, freq="D", tz="UTC")
    return pd.DataFrame({"close": [100.0, 100.0, 100.0]}, index=idx)

if __name__ == "__main__":
    df = build_dummy_df()
    strat = DummyMLStrategy()
    res = strat.run(df, run_config={"cash": 100000.0})

    equity = res["equity"]
    trades = res["trades"]

    # Print required confirmations
    print(f"Equity: name={equity.name}, len={len(equity)}, tz={equity.index.tz}")
    print(f"Trades cols: {list(trades.columns)}  rows={len(trades)}")

    # Tail hint: see README, logs folder
    print("OK: sanity_ml_hooks finished.")

