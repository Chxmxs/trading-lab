# -*- coding: utf-8 -*-
"""
entry_filter.py — veto low-probability trades using an ML model.
If LightGBM is not installed, acts as a no-op passthrough.
Expected inputs:
  - entries_df: DataFrame of candidate entries with at least ["timestamp", ...]
  - feats_df:   DataFrame of features aligned by timestamp
"""
from __future__ import annotations
from typing import Optional
import pandas as pd

def _merge_on_ts(entries_df: pd.DataFrame, feats_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if feats_df is None or feats_df.empty:
        return entries_df.copy()
    if "timestamp" not in entries_df.columns or "timestamp" not in feats_df.columns:
        return entries_df.copy()
    e = entries_df.copy()
    f = feats_df.copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True, errors="coerce")
    e["timestamp"] = pd.to_datetime(e["timestamp"], utc=True, errors="coerce")
    return pd.merge_asof(e.sort_values("timestamp"), f.sort_values("timestamp"),
                         on="timestamp", direction="backward")

def filter_with_model(entries_df: pd.DataFrame,
                      feats_df: Optional[pd.DataFrame],
                      model_loader=None,
                      proba_col: str = "pred_proba",
                      threshold: float = 0.55) -> pd.DataFrame:
    merged = _merge_on_ts(entries_df, feats_df)
    if model_loader is None:
        # no-op: pass through
        merged[proba_col] = 0.6  # dummy "good"
        return merged[merged[proba_col] >= threshold].copy()

    # model_loader should return a callable: predict_proba(df)->array in [0,1]
    model = model_loader()
    try:
        proba = model.predict_proba(merged)  # user-defined
    except Exception:
        # If API differs, tolerate via attribute checks
        if hasattr(model, "predict"):
            proba = model.predict(merged)
        else:
            proba = [1.0] * len(merged)

    merged[proba_col] = proba
    return merged[merged[proba_col] >= threshold].copy()
