# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from companion.explorer.overlap import (
    load_trade_structure, interval_overlap_score, jaccard_points,
    load_master, prune_master_items, save_master
)

def _write_trades_points(p: Path, ts_list):
    pd.DataFrame({"ts": ts_list}).to_csv(p, index=False)

def _write_trades_intervals(p: Path, pairs):
    pd.DataFrame({"entry_ts":[a for a,_ in pairs],
                  "exit_ts":[b for _,b in pairs]}).to_csv(p, index=False)

def _write_equity(p: Path, ts, eq):
    pd.DataFrame({"ts": ts, "equity": eq}).to_csv(p, index=False)

def test_overlap_and_pruning(tmp_path):
    # Build discrete trade points
    a_pts = [f"2020-01-01 00:0{i}:00Z" for i in range(5)]
    b_pts = [f"2020-01-01 00:0{i}:00Z" for i in range(3)] + ["2020-01-01 00:10:00Z","2020-01-01 00:11:00Z"]

    a_trades = tmp_path / "a_trades.csv"
    b_trades = tmp_path / "b_trades.csv"
    _write_trades_points(a_trades, a_pts)
    _write_trades_points(b_trades, b_pts)

    a_e = tmp_path / "a_equity.csv"
    b_e = tmp_path / "b_equity.csv"
    # Highly correlated equity
    ts = [f"2020-01-01 00:0{i}:00Z" for i in range(6)]
    _write_equity(a_e, ts, [100,101,102,101,103,104])
    _write_equity(b_e, ts, [100,100.9,102,101,103,104])

    # Master list best-first: A first (better MAR), B second
    master = [
        {"run_id":"A","strategy":"S","symbol":"BTCUSD","timeframe":"15m",
         "metrics":{"mar":1.0,"sharpe":1.0,"cagr":0.5,"maxdd":0.2,"trades":5},
         "artifacts":{"trades_csv":str(a_trades),"equity_csv":str(a_e)}},
        {"run_id":"B","strategy":"S","symbol":"BTCUSD","timeframe":"15m",
         "metrics":{"mar":0.9,"sharpe":0.9,"cagr":0.4,"maxdd":0.25,"trades":5},
         "artifacts":{"trades_csv":str(b_trades),"equity_csv":str(b_e)}},
    ]

    # Sanity: Jaccard overlap is high (3/7 ~ 0.428), correlation high (~1.0)
    a_struct = load_trade_structure(str(a_trades))
    b_struct = load_trade_structure(str(b_trades))
    jac = jaccard_points(a_struct["points"], b_struct["points"])
    assert jac > 0.35

    # Prune with low thresholds -> B should be dropped
    result = prune_master_items(master, overlap_threshold=0.40, corr_threshold=0.80)
    kept = [x["run_id"] for x in result["kept"]]
    dropped = [x["run_id"] for x in result["dropped"]]
    assert kept == ["A"]
    assert "B" in dropped