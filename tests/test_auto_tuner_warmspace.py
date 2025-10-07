# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path
from companion.ml.auto_tuner import Policy, derive_warm_space, load_history, filter_key

def _make_trials(tmp_path: Path) -> Path:
    rows = []
    import numpy as np
    rng = np.random.default_rng(777)
    for i in range(120):
        rows.append({
            "strategy": "SOPRRegimeBand",
            "symbol": "BTCUSD",
            "timeframe": "15m",
            "trial_number": i,
            "end_time": f"2024-01-{(i%28)+1:02d}T00:00:00Z",
            "objective": float(rng.normal(0.8, 0.2) + 0.2*(i/120.0)),
            "param_band_low": float(rng.uniform(0.96, 0.99)),
            "param_band_high": float(rng.uniform(1.00, 1.06)),
            "param_ma_regime": int(rng.integers(50, 200)),
            "param_tp_pct": float(rng.uniform(0.02, 0.12))
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "trials.parquet"
    df.to_parquet(path)
    return path

def test_warm_space_derivation(tmp_path: Path):
    p = _make_trials(tmp_path)
    df = load_history([p])
    key = filter_key(df, "SOPRRegimeBand", "BTCUSD", "15m")
    pol = Policy(min_trials=60, top_frac=0.25)
    warm = derive_warm_space(key, pol)
    assert "param_band_low" in warm
    assert warm["param_band_low"]["low"] < warm["param_band_low"]["high"]
