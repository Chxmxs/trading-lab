# -*- coding: utf-8 -*-
import os
import pandas as pd
from companion.explorer.overlap import prune_overlap_strategies

def test_prune_overlap_basic(tmp_path):
    # Build candidate table
    df = pd.DataFrame([
        {"strategy_id":"A","symbol":"BTCUSD","timeframe":"15m","trades_csv":"artifacts/_ci/overlap/trades_A.csv","oos_mar":1.2},
        {"strategy_id":"B","symbol":"BTCUSD","timeframe":"15m","trades_csv":"artifacts/_ci/overlap/trades_B.csv","oos_mar":0.9},
        {"strategy_id":"C","symbol":"BTCUSD","timeframe":"15m","trades_csv":"artifacts/_ci/overlap/trades_C.csv","oos_mar":1.1},
    ])
    out = prune_overlap_strategies(df, overlap_threshold=0.5, score_col="oos_mar", mlflow_log=False, artifacts_dir=tmp_path/"overlap_artifacts")
    # A and B overlap strongly; A has higher score so B should be pruned; C should be kept
    assert "A" in out["kept"]
    assert "C" in out["kept"]
    assert "B" in out["pruned"]

    # Matrix & summary artifacts exist
    assert (tmp_path/"overlap_artifacts"/out["runstamp"]).exists()
