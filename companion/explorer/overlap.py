# -*- coding: utf-8 -*-
"""
companion.explorer.overlap
Phase 11: Overlap Pruning of strategies by trade-entry timestamp collision.
Deterministic, Windows-safe, no SciPy dependency (graph via pure Python).
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import mlflow
import pandas as pd

from companion.logging_config import get_logger  # assumes existing helper
from companion.common import utc_now_str  # assumes existing helper like '%Y-%m-%dT%H%M%S'
# If these helpers are missing, replace with local fallbacks.

logger = get_logger(__name__)

@dataclass(frozen=True)
class Candidate:
    strategy_id: str           # unique name, e.g. "SOPRRegimeBand@BTCUSD@15m@seed777"
    symbol: str                # e.g. "BTCUSD"
    timeframe: str             # e.g. "15m"
    score: float               # e.g. OOS MAR
    trades_csv: str            # path to trades CSV artifact (UTF-8)
    run_id: str | None = None  # optional MLflow run_id
    extra: Dict = None         # any extra fields

def _load_entry_timestamps(csv_path: str) -> pd.Index:
    """Load entry timestamps from a trades CSV (must have 'entry_time' col in UTC ISO or epoch)."""
    p = Path(csv_path)
    if not p.exists():
        logger.warning("Trades CSV not found: %s", csv_path)
        return pd.Index([], dtype='datetime64[ns, UTC]')
    df = pd.read_csv(p)
    col_candidates = [c for c in df.columns if c.lower() in ("entry_time","entry_timestamp","entry_dt","entry_at")]
    if not col_candidates:
        logger.warning("No entry_time-like column in %s (cols=%s)", csv_path, df.columns.tolist())
        return pd.Index([], dtype='datetime64[ns, UTC]')
    col = col_candidates[0]
    ts = pd.to_datetime(df[col], utc=True, errors="coerce")
    ts = ts.dropna()
    return pd.DatetimeIndex(ts).tz_convert("UTC")

def _jaccard_overlap(a: pd.Index, b: pd.Index) -> float:
    """Jaccard similarity on exact timestamps (bar-aligned)."""
    if len(a) == 0 and len(b) == 0:
        return 0.0
    set_a = set(a.view("int64"))
    set_b = set(b.view("int64"))
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0

def _build_overlap_matrix(cands: List[Candidate]) -> pd.DataFrame:
    """Return symmetric overlap matrix for candidates that share symbol+timeframe."""
    rows = []
    # Pre-load entries per candidate
    entries: Dict[str, pd.Index] = {}
    for c in cands:
        entries[c.strategy_id] = _load_entry_timestamps(c.trades_csv)

    ids = [c.strategy_id for c in cands]
    mat = pd.DataFrame(0.0, index=ids, columns=ids, dtype=float)

    # Compute pairwise overlaps within same (symbol,timeframe)
    for i, ci in enumerate(cands):
        for j in range(i, len(cands)):
            cj = cands[j]
            if (ci.symbol, ci.timeframe) != (cj.symbol, cj.timeframe):
                ov = 0.0
            else:
                ov = _jaccard_overlap(entries[ci.strategy_id], entries[cj.strategy_id])
            mat.iat[i, j] = ov
            mat.iat[j, i] = ov
            rows.append((ci.strategy_id, cj.strategy_id, ov))
    return mat

def _graph_components(ids: List[str], mat: pd.DataFrame, threshold: float) -> List[List[str]]:
    """Build an undirected graph where edge exists if overlap >= threshold. Return connected components."""
    # adjacency via dict of sets
    adj: Dict[str, set] = {i: set() for i in ids}
    for i_idx, i in enumerate(ids):
        for j_idx, j in enumerate(ids):
            if j_idx <= i_idx:
                continue
            if float(mat.at[i, j]) >= threshold:
                adj[i].add(j)
                adj[j].add(i)

    visited = set()
    comps: List[List[str]] = []
    for i in ids:
        if i in visited:
            continue
        stack = [i]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            stack.extend(list(adj[cur] - visited))
        comps.append(sorted(comp))
    return comps

def prune_overlap_strategies(
    candidates_df: pd.DataFrame,
    overlap_threshold: float = 0.60,
    score_col: str = "oos_mar",
    mlflow_log: bool = True,
    artifacts_dir: str | os.PathLike = "artifacts/overlap",
) -> Dict:
    """
    candidates_df columns (required):
      - strategy_id, symbol, timeframe, trades_csv, oos_mar (or custom score_col)
    Returns dict with kept/pruned lists and artifact paths.
    """
    required = {"strategy_id","symbol","timeframe","trades_csv", score_col}
    missing = required - set(map(str, candidates_df.columns))
    if missing:
        raise ValueError(f"candidates_df missing columns: {sorted(missing)}")

    # Partition by (symbol,timeframe) for correctness
    groups = list(candidates_df.groupby(["symbol","timeframe"], sort=True))
    runstamp = utc_now_str()
    outdir = Path(artifacts_dir) / runstamp
    outdir.mkdir(parents=True, exist_ok=True)

    kept_rows = []
    pruned_rows = []
    cluster_records = []

    for (sym, tf), g in groups:
        cands = [
            Candidate(
                strategy_id=row["strategy_id"],
                symbol=sym,
                timeframe=tf,
                score=float(row[score_col]),
                trades_csv=row["trades_csv"],
                run_id=row.get("run_id"),
                extra={}
            )
            for _, row in g.iterrows()
        ]
        if len(cands) <= 1:
            kept_rows.extend(g.to_dict("records"))
            cluster_records.append({
                "symbol": sym, "timeframe": tf,
                "clusters": [[cands[0].strategy_id]] if cands else []
            })
            continue

        mat = _build_overlap_matrix(cands)
        ids = [c.strategy_id for c in cands]
        comps = _graph_components(ids, mat, threshold=overlap_threshold)

        # Save matrix per (sym,tf)
        mat_path = outdir / f"overlap_{sym}_{tf}.csv"
        mat.to_csv(mat_path, index=True)

        # Within each component keep best by score; prune others
        scores = {row["strategy_id"]: float(row[score_col]) for _, row in g.iterrows()}
        for comp in comps:
            best = max(comp, key=lambda sid: scores[sid])
            kept = [r for _, r in g.iterrows() if r["strategy_id"] == best]
            kept_rows.extend(kept)
            discard = [sid for sid in comp if sid != best]
            pruned_rows.extend([r for _, r in g.iterrows() if r["strategy_id"] in discard])
        cluster_records.append({
            "symbol": sym, "timeframe": tf,
            "clusters": comps
        })

    kept_df = pd.DataFrame(kept_rows).drop_duplicates(subset=["strategy_id"]).reset_index(drop=True)
    pruned_df = pd.DataFrame(pruned_rows).drop_duplicates(subset=["strategy_id"]).reset_index(drop=True)

    # Write summary artifacts
    kept_path = outdir / "kept.csv";     kept_df.to_csv(kept_path, index=False)
    pruned_path = outdir / "pruned.csv"; pruned_df.to_csv(pruned_path, index=False)
    cluster_path = outdir / "clusters.json"; cluster_path.write_text(json.dumps(cluster_records, indent=2))

    summary = {
        "runstamp": runstamp,
        "overlap_threshold": overlap_threshold,
        "score_col": score_col,
        "kept": kept_df["strategy_id"].tolist(),
        "pruned": pruned_df["strategy_id"].tolist(),
        "artifacts_dir": str(outdir)
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    if mlflow_log:
        try:
            mlflow.set_experiment("explorer_overlap_pruning")
            with mlflow.start_run(run_name=f"overlap_prune_{runstamp}"):
                mlflow.log_params({
                    "overlap_threshold": overlap_threshold,
                    "score_col": score_col,
                    "kept_count": len(summary["kept"]),
                    "pruned_count": len(summary["pruned"]),
                })
                # Log all produced artifacts in this outdir
                for p in outdir.glob("*"):
                    mlflow.log_artifact(str(p))
                mlflow.set_tags({"phase":"11","module":"overlap"})
        except Exception as e:
            logger.warning("MLflow logging skipped due to error: %s", e)

    logger.info("Overlap pruning complete. Kept=%d, Pruned=%d, Artifacts=%s",
                len(summary["kept"]), len(summary["pruned"]), outdir)
    return summary
