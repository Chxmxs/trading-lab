# -*- coding: utf-8 -*-
"""
refresh_master.py — re-rank strategies, prune correlated ones, write leaderboard+master list.
Outputs:
  - leaderboard.md
  - master_list.json
"""
from __future__ import annotations
import json
from pathlib import Path
import mlflow
from datetime import datetime, timezone

from companion.logging_config import configure_logging

log = configure_logging(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_MD = REPO_ROOT / "leaderboard.md"
OUT_JSON = REPO_ROOT / "master_list.json"

def _get_runs():
    rows = []
    for exp in mlflow.search_experiments():
        df = mlflow.search_runs([exp.experiment_id], order_by=["metrics.MAR DESC"], max_results=1000)
        if df is None or len(df)==0: continue
        for _, r in df.iterrows():
            rows.append(r.to_dict())
    return rows

def _rank(rows):
    # basic rank by MAR, then Sharpe, then CAGR
    scored = []
    for r in rows:
        mar = float(r.get("metrics.MAR", 0) or 0)
        sh  = float(r.get("metrics.Sharpe", 0) or 0)
        cg  = float(r.get("metrics.CAGR", 0) or 0)
        tc  = int(r.get("metrics.TradeCount", 0) or 0)
        scored.append((mar, sh, cg, tc, r))
    scored.sort(key=lambda x: (-x[0], -x[1], -x[2], -x[3]))
    return [t[-1] for t in scored]

def _prune_overlap(rows):
    # if companion/explorer/overlap.py exists use it; else no-op
    try:
        from companion.explorer.overlap import prune_correlated  # expected helper
        return prune_correlated(rows)
    except Exception:
        return rows

def refresh():
    rows = _get_runs()
    ranked = _rank(rows)
    pruned = _prune_overlap(ranked)

    # master_list.json
    best = []
    for r in pruned[:200]:
        best.append({
            "run_id": r.get("run_id"),
            "experiment_id": r.get("experiment_id"),
            "strategy": r.get("tags.strategy", r.get("tags.strat", "")),
            "symbol": r.get("tags.symbol", ""),
            "timeframe": r.get("tags.timeframe", ""),
            "MAR": r.get("metrics.MAR", 0),
            "Sharpe": r.get("metrics.Sharpe", 0),
            "CAGR": r.get("metrics.CAGR", 0),
            "Trades": r.get("metrics.TradeCount", 0),
        })
    OUT_JSON.write_text(json.dumps(best, indent=2), encoding="utf-8")

    # leaderboard.md
    lines = []
    lines.append("# Strategy Leaderboard")
    lines.append("")
    lines.append(f"_Refreshed: {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")
    lines.append("| # | Strategy | Symbol | TF | MAR | Sharpe | CAGR | Trades | Run |")
    lines.append("|---|----------|--------|----|-----|--------|------|--------|-----|")
    for i, r in enumerate(pruned[:100], 1):
        lines.append(f"| {i} | {r.get('tags.strategy','')} | {r.get('tags.symbol','')} | {r.get('tags.timeframe','')} | "
                     f"{r.get('metrics.MAR',0):.3f} | {r.get('metrics.Sharpe',0):.3f} | {r.get('metrics.CAGR',0):.3f} | "
                     f"{int(r.get('metrics.TradeCount',0))} | `{r.get('run_id','')[:8]}` |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"ranked": len(ranked), "pruned": len(pruned), "json": str(OUT_JSON), "md": str(OUT_MD)}

if __name__ == "__main__":
    print(refresh())
