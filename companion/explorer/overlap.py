from __future__ import annotations
from pathlib import Path
from typing import Union, Iterable, Dict, Any
import os
import pandas as pd

__all__ = ["load_trade_structure", "jaccard_points", "prune_overlap_strategies"]

_CANON = ["run_id","strategy","symbol","timeframe","side","qty",
          "entry_time","entry_price","exit_time","exit_price",
          "pnl","trade_id","position_id"]

_COL_MAP = {
    "run":"run_id","runid":"run_id","strategy_name":"strategy","tf":"timeframe",
    "ticker":"symbol","asset":"symbol","side_name":"side","direction":"side",
    "quantity":"qty","size":"qty","entry":"entry_time","entry_ts":"entry_time",
    "open_time":"entry_time","entryprice":"entry_price","open_price":"entry_price",
    "exit":"exit_time","exit_ts":"exit_time","close_time":"exit_time",
    "exitprice":"exit_price","close_price":"exit_price","profit":"pnl",
    "pnl_quote":"pnl","pnl_usd":"pnl","id":"trade_id","position":"position_id",
}

def _to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # rename common variants
    ren = {}
    for c in df.columns:
        k = str(c).strip().lower()
        ren[c] = _COL_MAP.get(k, k)
    df = df.rename(columns=ren)

    # sides
    if "side" in df.columns:
        df["side"] = (
            df["side"]
            .astype(str)
            .str.upper()
            .map({"BUY":"LONG","LONG":"LONG","L":"LONG","SELL":"SHORT","SHORT":"SHORT","S":"SHORT"})
            .fillna(df["side"])
        )

    # times
    for t in ("entry_time","exit_time"):
        if t in df.columns:
            df[t] = _to_utc(df[t])

    # ensure required cols exist
    defaults = {"run_id":"","strategy":"","symbol":"","timeframe":"","side":"",
                "qty":0.0,"entry_time":pd.NaT,"entry_price":float("nan"),
                "exit_time":pd.NaT,"exit_price":float("nan"),"pnl":0.0,
                "trade_id":"","position_id":""}
    for k,v in defaults.items():
        if k not in df.columns:
            df[k] = v

    extras = [c for c in df.columns if c not in _CANON]
    df = df[_CANON + extras]
    if "entry_time" in df.columns:
        df = df.sort_values("entry_time", kind="stable", na_position="last")
    return df

def load_trade_structure(source: Union[str, os.PathLike, pd.DataFrame], *, assume_csv_has_header: bool = True) -> pd.DataFrame:
    """Load trades from CSV or DataFrame and normalize schema. Surface 'points' from 'ts' if present."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Trades file not found: {p}")
        kw = {}
        if not assume_csv_has_header:
            kw["header"] = None
        df = pd.read_csv(p, **kw)
    df = _normalize(df)
    # tests use a CSV with a 'ts' column of discrete trade timestamps
    if "ts" in df.columns and "points" not in df.columns:
        df["points"] = df["ts"]
    return df

def jaccard_points(a: Iterable[Any], b: Iterable[Any]) -> float:
    """Jaccard index on point sets (timestamps or strings)."""
    import pandas as pd
    def _norm(seq):
        S = set()
        if seq is None: return S
        for v in seq:
            if v is None: continue
            try:
                t = pd.to_datetime(v, utc=True)
                if pd.isna(t): continue
                S.add(int(t.value))  # ns epoch
            except Exception:
                S.add(str(v))
        return S
    A = _norm(a); B = _norm(b)
    if not A and not B: return 0.0
    return float(len(A & B) / len(A | B))

def prune_overlap_strategies(df: pd.DataFrame, *,
                             overlap_threshold: float = 0.75,
                             score_col: str = "oos_mar",
                             mlflow_log: bool = False,
                             artifacts_dir = None,
                             threshold: float = None,
                             score_column: str = None) -> pd.DataFrame:
    """
    Prune a candidate master list of strategies by overlap.
    Expected input (per tests): rows with columns including 'strategy_id' and 'trades_csv',
    optionally a score column (default 'oos_mar'). Keep best-first, drop later ones whose
    Jaccard(points) with any kept >= overlap_threshold.
    """
    # Accept alias names (if tests call with threshold/score_column instead)
    if threshold is not None:
        overlap_threshold = float(threshold)
    if score_column is not None:
        score_col = str(score_column)

    # If this already looks like per-trade rows (has entry_time), no-op
    if "entry_time" in df.columns and "trades_csv" not in df.columns:
        return df

    # Build points per candidate
    rows = []
    for _, r in df.iterrows():
        sid = str(r.get("strategy_id", r.get("strategy", "")))
        tcsv = r.get("trades_csv")
        if pd.isna(tcsv) or not tcsv:
            rows.append((sid, set(), r))
            continue
        tdf = load_trade_structure(str(tcsv))
        pts = set()
        if "points" in tdf.columns:
            pts = set(tdf["points"].dropna().astype(str).tolist())
        elif "entry_time" in tdf.columns:
            pts = set(tdf["entry_time"].dropna().astype(str).tolist())
        rows.append((sid, pts, r))

    # Sort candidates best-first by score_col (desc), missing -> very low
    def _score(rr):
        v = rr[2].get(score_col, None)
        try:
            return float(v)
        except Exception:
            return float("-inf")
    rows.sort(key=_score, reverse=True)

    kept_ids = []
    kept_pts = []
    kept_rows = []

    for sid, pts, meta in rows:
        drop = False
        for other_pts in kept_pts:
            if not pts and not other_pts:
                continue
            jac = jaccard_points(pts, other_pts)
            if jac >= float(overlap_threshold):
                drop = True
                break
        if not drop:
            kept_ids.append(sid)
            kept_pts.append(pts)
            kept_rows.append(meta)

    # Return a DataFrame of kept rows (preserve original columns)
    if kept_rows:
        return pd.DataFrame(kept_rows).reset_index(drop=True)
    return df.iloc[0:0].copy()

def interval_overlap_score(a_start, a_end, b_start, b_end) -> float:
    import pandas as pd
    a0 = pd.to_datetime(a_start, utc=True); a1 = pd.to_datetime(a_end, utc=True)
    b0 = pd.to_datetime(b_start, utc=True); b1 = pd.to_datetime(b_end, utc=True)
    if a1 < a0: a0, a1 = a1, a0
    if b1 < b0: b0, b1 = b1, b0
    inter_start = max(a0, b0); inter_end = min(a1, b1)
    inter = (inter_end - inter_start).total_seconds()
    if inter <= 0:
        return 0.0
    dur_a = (a1 - a0).total_seconds()
    dur_b = (b1 - b0).total_seconds()
    union = dur_a + dur_b - inter
    return float(inter / union) if union > 0 else 0.0
