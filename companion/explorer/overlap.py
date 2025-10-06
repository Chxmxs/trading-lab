from __future__ import annotations
from pathlib import Path
from typing import Union
import os, pandas as pd

__all__ = ["load_trade_structure"]

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
        df["side"] = df["side"].astype(str).str.upper().map(
            {"BUY":"LONG","LONG":"LONG","L":"LONG","SELL":"SHORT","SHORT":"SHORT","S":"SHORT"}
        ).fillna(df["side"])

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
    """Load trades from CSV or DataFrame and normalize schema."""
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
    if "ts" in df.columns and "points" not in df.columns:
        df["points"] = df["ts"]  # test fixtures expect this alias
    return df
