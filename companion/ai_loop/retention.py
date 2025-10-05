# -*- coding: utf-8 -*-
"""
retention.py — keep best N runs per strategy; archive older artifacts.
"""
from __future__ import annotations
from pathlib import Path
import shutil
import mlflow

from companion.logging_config import configure_logging
log = configure_logging(__name__)

ARCHIVE_ROOT = Path("artifacts/_archive")
ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

def archive_path(run_id: str) -> Path:
    return ARCHIVE_ROOT / run_id

def apply_retention(per_strategy: int = 5):
    # fetch runs grouped by strategy tag
    by_strat = {}
    for exp in mlflow.search_experiments():
        df = mlflow.search_runs([exp.experiment_id], order_by=["metrics.MAR DESC"], max_results=10000)
        if df is None or len(df)==0: continue
        for _, r in df.iterrows():
            strat = r.get("tags.strategy", "UNKNOWN")
            by_strat.setdefault(strat, []).append(r)

    # for each, keep top N by MAR; archive others
    for strat, rows in by_strat.items():
        rows.sort(key=lambda r: float(r.get("metrics.MAR",0) or 0), reverse=True)
        keep = set([r.get("run_id") for r in rows[:per_strategy]])
        for r in rows[per_strategy:]:
            rid = r.get("run_id")
            src = Path("artifacts") / rid if rid else None
            if src and src.exists():
                dst = archive_path(rid)
                if not dst.exists():
                    try:
                        shutil.move(str(src), str(dst))
                        log.info("Archived %s for strategy %s", rid, strat)
                    except Exception as e:
                        log.error("Archive failed %s: %s", rid, e)
