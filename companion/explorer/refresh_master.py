# -*- coding: utf-8 -*-
"""
companion.explorer.refresh_master
Integrates overlap pruning before writing master list / leaderboard.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from companion.explorer.overlap import prune_overlap_strategies
from companion.logging_config import get_logger

logger = get_logger(__name__)

def refresh_master(candidates_csv: str = "leaderboard_candidates.csv",
                   master_out: str = "master_list.json",
                   leaderboard_md: str = "leaderboard.md",
                   overlap_threshold: float = 0.60,
                   score_col: str = "oos_mar") -> None:
    if not Path(candidates_csv).exists():
        raise FileNotFoundError(f"Missing candidates file: {candidates_csv}")

    df = pd.read_csv(candidates_csv)
    # Expect df to include: strategy_id, symbol, timeframe, trades_csv, oos_mar (or score_col)
    out = prune_overlap_strategies(df, overlap_threshold=overlap_threshold, score_col=score_col)

    kept_df = df[df["strategy_id"].isin(out["kept"])].copy()
    kept_df.sort_values(by=[score_col], ascending=False, inplace=True)
    # Persist master list
    Path(master_out).write_text(json.dumps(kept_df.to_dict(orient="records"), indent=2))
    # Minimal leaderboard markdown
    lines = ["# Leaderboard (Post-Overlap Pruning)", "", f"- Kept: {len(out['kept'])}", f"- Pruned: {len(out['pruned'])}", ""]
    lines.append("| Rank | Strategy | Symbol | TF | Score |")
    lines.append("|------|----------|--------|----|-------|")
    for i, r in enumerate(kept_df.itertuples(index=False), start=1):
        lines.append(f"| {i} | {r.strategy_id} | {r.symbol} | {r.timeframe} | {getattr(r, score_col):.4f} |")
    Path(leaderboard_md).write_text("\n".join(lines))

    logger.info("Master list refreshed with overlap pruning: kept=%d, pruned=%d", len(out["kept"]), len(out["pruned"]))
