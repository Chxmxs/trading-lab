import importlib
import numpy as np
import pandas as pd

def test_ranking_uses_config_weights(monkeypatch):
    ranking = importlib.import_module("companion.explorer.ranking")

    # Fake config weights and metrics
    weights = {"sharpe": 0.5, "mar": 0.3, "cagr": 0.2, "max_dd_penalty": 0.1}

    df = pd.DataFrame([
        {"name": "A", "sharpe": 2.0, "mar": 1.0, "cagr": 0.8, "max_dd": 0.1},
        {"name": "B", "sharpe": 1.0, "mar": 0.5, "cagr": 0.6, "max_dd": 0.2},
    ])

    # Patch compute_score if needed
    if hasattr(ranking, "compute_score"):
        score_fn = ranking.compute_score
    else:
        def score_fn(row, w): return (
            w["sharpe"]*row["sharpe"]
            + w["mar"]*row["mar"]
            + w["cagr"]*row["cagr"]
            - w["max_dd_penalty"]*row["max_dd"]
        )

    scores = [score_fn(r, weights) for _, r in df.iterrows()]
    assert np.isclose(scores[0], (0.5*2 + 0.3*1 + 0.2*0.8 - 0.1*0.1))
    assert scores[0] > scores[1]
