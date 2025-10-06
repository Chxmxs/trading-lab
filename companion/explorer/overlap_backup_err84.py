from __future__ import annotations
import os
from pathlib import Path
from typing import Union
import pandas as pd

__all__ = ["load_trade_structure","interval_overlap_score","compute_overlap_matrix","prune_overlap_strategies"]

_CANON = [
    "run_id","strategy","symbol","timeframe","side","qty",
    "entry_time","entry_price","exit_time","exit_price","pnl",
    "trade_id","position_id",
]

_COL_MAP = {
    "run":"run_id","runid":"run_id","strategy_name":"strategy","tf":"timeframe",
    "ticker":"symbol","asset":"symbol","side_name":"side","direction":"side",
    "quantity":"qty","size":"qty","entry":"entry_time","entry_ts":"entry_time",
    "open_time":"entry_time","entryprice":"entry_price","open_price":"entry_price",
    "exit":"exit_time","exit_ts":"exit_time","close_time":"exit_time",
    "exitprice":"exit_price","close_price":"exit_price","profit":"pnl",
    "pnl_quote":"pnl","pnl_usd":"pnl","id":"trade_id","position":"position_id",
}

def _to_utc(ts: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(ts):
        s = pd.to_datetime(ts, utc=True)
    else:
        s = pd.to_datetime(ts, errors="coerce", utc=True)
    return s

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        key = str(c).strip()
        lower = key.lower()
        ren[c] = _COL_MAP.get(lower, lower)
    df = df.rename(columns=ren)
    if "side" in df.columns:
        def _norm_side(x):
            x = str(x).upper()
            if x in ("BUY","LONG","L"): return "LONG"
            if x in ("SELL","SHORT","S"): return "SHORT"
            return x
        df["side"] = df["side"].map(_norm_side)
    for tcol in ("entry_time","exit_time"):
        if tcol in df.columns:
            df[tcol] = _to_utc(df[tcol])
    defaults = {
        "run_id":"","strategy":"","symbol":"","timeframe":"",
        "side":"","qty":0.0,"entry_time":pd.NaT,"entry_price":float("nan"),
        "exit_time":pd.NaT,"exit_price":float("nan"),"pnl":0.0,
        "trade_id":"","position_id":"",
    }
    for c,v in defaults.items():
        if c not in df.columns:
            df[c] = v
    extras = [c for c in df.columns if c not in _CANON]
    df = df[_CANON + extras]
    if "entry_time" in df.columns:
        df = df.sort_values("entry_time", kind="stable", na_position="last")
    return df

def load_trade_structure(source: Union[str, os.PathLike, pd.DataFrame], assume_csv_has_header: bool = True) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError("Trades file not found: %s" % path)
        read_kwargs = {}
        if not assume_csv_has_header:
            read_kwargs["header"] = None
        df = pd.read_csv(path, **read_kwargs)
    return _normalize_columns(df)

def interval_overlap_score(a_start, a_end, b_start, b_end) -> float:
    import pandas as pd
    a0 = pd.to_datetime(a_start, utc=True); a1 = pd.to_datetime(a_end, utc=True)
    b0 = pd.to_datetime(b_start, utc=True); b1 = pd.to_datetime(b_end, utc=True)
    if a1 < a0: a0, a1 = a1, a0
    if b1 < b0: b0, b1 = b1, b0
    inter_start = max(a0, b0); inter_end = min(a1, b1)
    inter = (inter_end - inter_start).total_seconds()
    if inter <= 0: return 0.0
    dur_a = (a1 - a0).total_seconds()
    dur_b = (b1 - b0).total_seconds()
    union = dur_a + dur_b - inter
    return float(inter / union) if union > 0 else 0.0

def compute_overlap_matrix(df):
    import pandas as pd
    req = {"strategy","entry_time","exit_time"}
    missing = req - set(df.columns)
    if missing: raise ValueError("missing columns: %s" % sorted(missing))
    strategies = sorted(df["strategy"].dropna().astype(str).unique().tolist())
    mat = pd.DataFrame(0.0, index=strategies, columns=strategies)
    counts = pd.DataFrame(0, index=strategies, columns=strategies)
    grouped = {s: g[["entry_time","exit_time"]].reset_index(drop=True) for s,g in df.groupby("strategy", dropna=True)}
    for i, si in enumerate(strategies):
        gi = grouped.get(si)
        for j, sj in enumerate(strategies):
            gj = grouped.get(sj)
            if gi is None or gj is None:
                continue
            total = 0.0; n = 0
            for _, ai in gi.iterrows():
                for _, bj in gj.iterrows():
                    s = interval_overlap_score(ai["entry_time"], ai["exit_time"], bj["entry_time"], bj["exit_time"])
                    if s > 0:
                        total += s; n += 1
            if n > 0:
                mat.loc[si, sj] = total / float(n)
                counts.loc[si, sj] = n
    return mat, counts

def prune_overlap_strategies(df, *, threshold: float = 0.75, score_column: str = "pnl"):
    import pandas as pd
    if df.empty:
        return df
    mat, _ = compute_overlap_matrix(df)
    totals = df.groupby("strategy")[score_column].sum().sort_values(ascending=False)
    keep = []
    banned = set()
    for s in totals.index:
        if s in banned:
            continue
        keep.append(s)
        overlaps = mat.loc[s]
        too_close = overlaps[overlaps >= threshold].index.tolist()
        for t in too_close:
            if t == s:
                continue
            banned.add(t)
    return df[df["strategy"].isin(keep)].reset_index(drop=True)
