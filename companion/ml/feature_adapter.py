# -*- coding: utf-8 -*-
"""
feature_adapter.py — parquet-based feature store.
Paths: data/parquet/{symbol}/{tf}/{feature_name}.parquet
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "data" / "parquet"

def feature_path(symbol: str, timeframe: str, name: str) -> Path:
    return ROOT / symbol / timeframe / f"{name}.parquet"

def save_features(df: pd.DataFrame, symbol: str, timeframe: str, name: str) -> Path:
    p = feature_path(symbol, timeframe, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p

def load_features(symbol: str, timeframe: str, name: str) -> Optional[pd.DataFrame]:
    p = feature_path(symbol, timeframe, name)
    if not p.exists(): return None
    return pd.read_parquet(p)

def list_features(symbol: str, timeframe: str) -> Dict[str, Path]:
    base = ROOT / symbol / timeframe
    out: Dict[str, Path] = {}
    if base.exists():
        for f in base.glob("*.parquet"):
            out[f.stem] = f
    return out
