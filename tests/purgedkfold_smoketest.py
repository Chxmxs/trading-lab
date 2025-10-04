# tests/purgedkfold_smoketest.py
# Purpose: prove PurgedKFold+embargo works without PyBroker, using synthetic data
# and using the same multi-path resolver that evaluation.py uses.

from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from companion.explorer.evaluation import evaluate_strategy_cv, CVConfig, PurgedKFold as PKF

# --- tiny dummy "strategy" that returns an equity curve and fake trades ---
def dummy_strategy(df: pd.DataFrame, run_config: dict) -> dict:
    px = df["price"].copy()
    mom = np.sign(px.diff(3).fillna(0.0))
    ret = 0.0002 + mom * 0.001
    equity = (1.0 + ret).cumprod() * 100.0
    equity.index = df.index
    trades = pd.DataFrame({"entry_time": df.index[::25]})
    return {"equity": equity, "trades": trades}

def build_synthetic_data(n=1200, seed=777) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0003, scale=0.01, size=n)
    price = 100 * np.exp(np.cumsum(steps))
    idx = pd.date_range("2024-01-01 00:00:00+00:00", periods=n, freq="1min")
    return pd.DataFrame({"price": price}, index=idx)

def main():
    # Use the same PurgedKFold object that evaluation.py resolved
    if PKF is not None:
        print("✅ mlfinlab detected via evaluation.PurgedKFold; PurgedKFold will be used.")
    else:
        print("ℹ️ mlfinlab NOT available via evaluation; test will fall back to KFold/sequential.")

    data = build_synthetic_data()
    cv = CVConfig(n_splits=5, embargo_pct=0.02)
    metrics = evaluate_strategy_cv(
        strategy_func=dummy_strategy,
        data=data,
        run_config={},
        cv_config=cv,
        timestamp_column="timestamp",
    )

    print("\n=== Aggregated CV Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Optional: preview fold test ranges if PKF is available
    if PKF is not None:
        try:
            data_idx = data.index
            emb_frac = cv.embargo_pct
            emb_rows = int(emb_frac * len(data_idx))
            pkf = None
            try:
                pkf = PKF(n_splits=cv.n_splits, samples_info_sets=data_idx, embargo_td=emb_rows)
            except Exception:
                pkf = PKF(n_splits=cv.n_splits, samples_info_sets=data_idx, pct_embargo=emb_frac)
            print("\n=== PurgedKFold splits (test start/end) ===")
            for i, (tr, te) in enumerate(pkf.split(None)):
                print(f"Fold {i}: test [{data_idx[te[0]]} .. {data_idx[te[-1]]}] len={len(te)}")
        except Exception as e:
            print("Split preview unavailable:", e)

if __name__ == "__main__":
    main()
