# -*- coding: utf-8 -*-
"""
feature_adapter.py — Parquet-based feature & label accessors.
Paths:
  Features: data/parquet/{symbol}/{tf}/{feature_name}.parquet
  Labels:   data/parquet/{symbol}/{tf}/labels_*.parquet
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

# Repo-aware root so this works no matter the CWD
REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "data" / "parquet"

# ------------------------------------------------------------------------------
# Basic, price-derived features (fallback if you don't precompute richer ones)
# ------------------------------------------------------------------------------

def basic_price_features_from_csv(price_csv: str, *, tf: str = "", max_lag: int = 5) -> pd.DataFrame:
    """
    Build a tiny feature set from a price CSV with cols: timestamp/ts, open, high, low, close, volume.
    Features: lagged returns, rolling volatility.
    Returns a DataFrame indexed by timestamp (UTC).
    """
    import numpy as np

    df = pd.read_csv(price_csv)
    # Detect timestamp column
    ts_col = "timestamp" if "timestamp" in df.columns else "ts" if "ts" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.set_index(ts_col).sort_index()

    # Detect close column
    if "close" not in df.columns:
        close_like = [c for c in df.columns if c.lower().startswith("close")]
        if not close_like:
            raise ValueError("CSV missing 'close' column")
        close_col = close_like[0]
    else:
        close_col = "close"

    ret1 = df[close_col].pct_change()
    feats = {"ret1": ret1}
    for k in range(2, max_lag + 1):
        feats[f"ret{k}"] = df[close_col].pct_change(k)
    feats["vol5"] = ret1.rolling(5).std()

    feats_df = pd.DataFrame(feats).replace([np.inf, -np.inf], pd.NA).fillna(0.0)
    feats_df.index = pd.DatetimeIndex(feats_df.index).tz_convert("UTC")
    return feats_df

# ------------------------------------------------------------------------------
# Feature store helpers
# ------------------------------------------------------------------------------

def feature_path(symbol: str, timeframe: str, name: str) -> Path:
    return ROOT / str(symbol) / str(timeframe) / f"{name}.parquet"

def save_features(df: pd.DataFrame, symbol: str, timeframe: str, name: str) -> Path:
    """
    Save features, **preserving the DateTimeIndex**. This is important for alignment.
    """
    p = feature_path(symbol, timeframe, name)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Ensure DateTimeIndex (UTC) if possible
    if not isinstance(df.index, pd.DatetimeIndex):
        # try to promote a timestamp column to index if present
        cand = None
        for c in ("timestamp", "ts"):
            if c in df.columns:
                cand = c
                break
        if cand is not None:
            df = df.copy()
            df[cand] = pd.to_datetime(df[cand], utc=True, errors="coerce")
            df = df.set_index(cand).sort_index()

    if isinstance(df.index, pd.DatetimeIndex):
        # force UTC for consistency
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    df.to_parquet(p)  # keep index!
    return p

def load_features(symbol: str, timeframe: str, name: str) -> Optional[pd.DataFrame]:
    p = feature_path(symbol, timeframe, name)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    # Normalize index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], utc=True)).drop(columns=["timestamp"])
        else:
            # last resort: try to coerce index to datetime
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception:
                pass
    return df

def list_features(symbol: str, timeframe: str) -> Dict[str, Path]:
    base = ROOT / str(symbol) / str(timeframe)
    out: Dict[str, Path] = {}
    if base.exists():
        for f in base.glob("*.parquet"):
            out[f.stem] = f
    return out

# ------------------------------------------------------------------------------
# Label accessors (Step 6 deliverable)
# ------------------------------------------------------------------------------

def _labels_dir(symbol: str, timeframe: str, root: Path = ROOT) -> Path:
    return root / str(symbol) / str(timeframe)

def find_latest_labels(symbol: str, timeframe: str, root: str | Path = ROOT) -> Path:
    """
    Locate the newest labels parquet for a symbol@timeframe.
    """
    d = _labels_dir(symbol, timeframe, Path(root))
    if not d.exists():
        raise FileNotFoundError(f"No label dir: {d}")
    files = sorted(d.glob("labels_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No labels_*.parquet in: {d}")
    return files[0]

def load_latest_labels(symbol: str, timeframe: str, root: str | Path = ROOT) -> pd.DataFrame:
    """
    Load the newest labels parquet for a symbol@timeframe from data/parquet/<sym>/<tf>/labels_*.parquet.
    Ensures a UTC DateTimeIndex and at least columns ['bin','ret','t1'] when present.
    """
    p = find_latest_labels(symbol, timeframe, root)
    df = pd.read_parquet(p)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        # Some writers put 'timestamp' as a column
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], utc=True)).drop(columns=["timestamp"])
        else:
            # if 't1' exists but index is range, still set index to DatetimeIndex of existing index (not ideal)
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception:
                raise ValueError(f"Labels parquet lacks datetime index and 'timestamp' column: {p}")

    # Normalize t1 to UTC if present
    if "t1" in df.columns:
        try:
            df["t1"] = pd.to_datetime(df["t1"], utc=True)
        except Exception:
            pass

    return df

def join_features_with_labels(features_df: pd.DataFrame, labels_df: pd.DataFrame, how: str = "inner") -> pd.DataFrame:
    """
    Convenience: align features with labels on timestamp index.
    Adds the 'bin' target column from labels_df.
    """
    if not isinstance(features_df.index, pd.DatetimeIndex):
        raise ValueError("features_df must have a DatetimeIndex")
    if not isinstance(labels_df.index, pd.DatetimeIndex):
        raise ValueError("labels_df must have a DatetimeIndex")

    cols = [c for c in ("bin",) if c in labels_df.columns]
    if not cols:
        raise ValueError("labels_df must contain a 'bin' column for meta-labeling")
    return features_df.join(labels_df[cols], how=how)
