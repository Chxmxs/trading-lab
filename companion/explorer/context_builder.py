# -*- coding: utf-8 -*-
"""
context_builder.py — build a data map from cleaned OHLCV, metrics CSVs, and parquet feature sets.
Scans:
  data/cleaned   -> ohlcv_{SYMBOL}@{TF}.csv
  data/metrics   -> metric_{slug}@{TF}.csv
  data/parquet   -> {symbol}/{tf}/*.parquet  (feature store)
Writes: companion/explorer/data_map.json
"""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DIR_CLEANED = REPO_ROOT / "data" / "cleaned"
DIR_METRICS = REPO_ROOT / "data" / "metrics"
DIR_PARQUET = REPO_ROOT / "data" / "parquet"
DATA_MAP_PATH = Path(__file__).resolve().parent / "data_map.json"

def _parse_ohlcv_filename(fname: str) -> Optional[Tuple[str,str]]:
    if not (fname.startswith("ohlcv_") and "@" in fname and fname.endswith(".csv")):
        return None
    stem = fname[:-4]; left, tf = stem.split("@", 1)
    symbol = left.split("_", 1)[-1]; return symbol, tf

def _parse_metric_filename(fname: str) -> Optional[Tuple[str,str,str]]:
    if not (fname.startswith("metric_") and "@" in fname and fname.endswith(".csv")):
        return None
    stem = fname[:-4]; slug, tf = stem.split("@", 1)
    s = slug.lower()
    if s.startswith("metric_btc"): sym = "BTC"
    elif s.startswith("metric_eth"): sym = "ETH"
    else: sym = "GLOBAL"
    return sym, tf, slug

def _add(lst: List[str], item: str):
    if item not in lst: lst.append(item)

def build_data_map() -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """
    Returns: symbol -> timeframe -> {
      'ohlcv': [extra_cols],
      'metrics': [metric_slugs],
      'parquet': [feature_names (files)]
    }
    Only uses data/cleaned, data/metrics, data/parquet.
    """
    dm: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    # OHLCV
    if DIR_CLEANED.exists():
        for fname in os.listdir(DIR_CLEANED):
            p = _parse_ohlcv_filename(fname)
            if not p: continue
            sym, tf = p; path = DIR_CLEANED / fname
            try:
                df = pd.read_csv(path, nrows=1)
                extra = [c for c in df.columns if c not in ("timestamp","open","high","low","close","volume")]
            except Exception:
                extra = []
            dm.setdefault(sym, {}).setdefault(tf, {})["ohlcv"] = extra

    # Metrics
    if DIR_METRICS.exists():
        for fname in os.listdir(DIR_METRICS):
            p = _parse_metric_filename(fname)
            if not p: continue
            sym, tf, slug = p
            dm.setdefault(sym, {}).setdefault(tf, {}).setdefault("metrics", [])
            _add(dm[sym][tf]["metrics"], slug)

    # Parquet feature store: data/parquet/{symbol}/{tf}/*.parquet
    if DIR_PARQUET.exists():
        for sym_dir in DIR_PARQUET.iterdir():
            if not sym_dir.is_dir(): continue
            sym = sym_dir.name
            for tf_dir in sym_dir.iterdir():
                if not tf_dir.is_dir(): continue
                tf = tf_dir.name
                feats: List[str] = []
                for f in tf_dir.glob("*.parquet"):
                    feats.append(f.stem)
                if feats:
                    dm.setdefault(sym, {}).setdefault(tf, {})["parquet"] = sorted(feats)

    DATA_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_MAP_PATH.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(dm, f, indent=2)
    return dm

if __name__ == "__main__":
    print(json.dumps(build_data_map(), indent=2))
