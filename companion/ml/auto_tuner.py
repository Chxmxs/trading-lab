# -*- coding: utf-8 -*-
"""
companion.ml.auto_tuner
Learns from historical tuning trials to propose better param regions (Step 3).
- Input: CSV with columns: objective (e.g., oos_mar) + param_* columns.
- Model: LightGBM if available, else RandomForestRegressor.
- Output artifacts: feature_importances.csv, suggestions.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd

from companion.logging_config import get_logger
from companion.common import utc_now_str

logger = get_logger(__name__)

try:
    from lightgbm import LGBMRegressor  # optional
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

@dataclass
class TunerConfig:
    objective_col: str = "oos_mar"
    top_k: int = 10             # how many top seeds to emit
    bound_pct: float = 0.2      # 20% band around top quantiles for numeric params
    experiment: str = "adaptive_tuner"
    artifacts_dir: str = "artifacts/auto_tuner"

def _split_features(df: pd.DataFrame, objective_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    y = df[objective_col].astype(float)
    param_cols = [c for c in df.columns if c != objective_col]
    X = df[param_cols].copy()
    # Coerce categorical/text to codes safely
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = X[c].astype("category").cat.codes
    return X, y, param_cols

def _train_model(X: pd.DataFrame, y: pd.Series):
    if _HAS_LGBM:
        model = LGBMRegressor(n_estimators=400, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=777)
    else:
        model = RandomForestRegressor(n_estimators=400, random_state=777, n_jobs=-1)
    model.fit(X, y)
    return model

def _feature_importances(model, cols: List[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        df = pd.DataFrame({"feature": cols, "importance": imps}).sort_values("importance", ascending=False).reset_index(drop=True)
        return df
    return pd.DataFrame({"feature": cols, "importance": [np.nan]*len(cols)})

def _numeric_bounds(series: pd.Series, pct: float) -> Tuple[float, float]:
    lo = float(series.quantile(pct))
    hi = float(series.quantile(1.0 - pct))
    if lo > hi: lo, hi = hi, lo
    return lo, hi

def suggest_from_trials(trials_csv: str, cfg: TunerConfig = TunerConfig()) -> Dict:
    df = pd.read_csv(trials_csv)
    if df.shape[0] < 5:
        raise ValueError("Need at least 5 trials to learn suggestions.")
    X, y, cols = _split_features(df, cfg.objective_col)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=777)

    mlflow.set_experiment(cfg.experiment)
    runstamp = utc_now_str()
    outdir = Path(cfg.artifacts_dir) / runstamp
    outdir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"auto_tuner_{runstamp}"):
        model = _train_model(Xtr, ytr)
        # Score (not critical; informative)
        try:
            r2 = float(model.score(Xte, yte))
        except Exception:
            r2 = float("nan")
        mlflow.log_metric("r2", r2)
        mlflow.log_param("model", model.__class__.__name__)
        mlflow.log_param("objective_col", cfg.objective_col)
        mlflow.log_param("top_k", cfg.top_k)
        mlflow.log_param("bound_pct", cfg.bound_pct)

        # Importance
        imps = _feature_importances(model, cols)
        imps_path = outdir / "feature_importances.csv"
        imps.to_csv(imps_path, index=False)
        mlflow.log_artifact(str(imps_path))

        # Rank by predicted objective
        preds = model.predict(X)
        df_ranked = df.copy()
        df_ranked["_pred"] = preds
        df_ranked.sort_values("_pred", ascending=False, inplace=True)
        top = df_ranked.head(cfg.top_k)

        # Bounds per numeric param based on top subset distribution
        suggestions = {"top_k_rows": top.drop(columns=["_pred"]).to_dict(orient="records"), "bounds": {}, "r2": r2}
        for c in cols:
            # only for numeric
            if np.issubdtype(df[c].dtype, np.number):
                lo, hi = _numeric_bounds(top[c], cfg.bound_pct)
                suggestions["bounds"][c] = {"low": lo, "high": hi}

        # Save & log
        sug_path = outdir / "suggestions.json"
        sug_path.write_text(json.dumps(suggestions, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(sug_path))
        mlflow.set_tags({"phase": "11", "module": "adaptive_tuner"})

    logger.info("Adaptive tuner complete. r2=%.4f, artifacts=%s", r2, outdir)
    return {"runstamp": runstamp, "artifacts_dir": str(outdir), "r2": r2}
