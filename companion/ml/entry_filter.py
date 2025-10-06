# -*- coding: utf-8 -*-
"""
Entry filter: load an ML model, score candidate entries, and keep those above a threshold.

Supports:
- MLflow model URIs: runs:/..., models:/...
- Local .pkl (joblib) models
- Local LightGBM .txt models (Booster)
- Direct 'model' objects passed in

Return shape:
{
  "entries": <filtered DataFrame or original if X is None>,
  "mask":    <pd.Series[bool] aligned to X.index or entries.index>,
  "proba":   <pd.Series[float] same index>,
  "threshold": float,
  "stats":   {"kept": int, "total": int, "accept_rate": float}
}
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

def _is_mlflow_uri(uri: str) -> bool:
    s = str(uri).lower()
    return s.startswith("runs:") or s.startswith("models:")

def _load_mlflow_model(uri: str):
    import mlflow
    return mlflow.pyfunc.load_model(uri)

def _load_joblib(path: str):
    import joblib
    return joblib.load(path)

def _load_lightgbm_booster(path: str):
    import lightgbm as lgb
    booster = lgb.Booster(model_file=path)
    class _BoosterWrapper:
        def __init__(self, booster):
            self.booster = booster
        def predict_proba(self, X):
            import numpy as np
            p = self.booster.predict(X)
            p = p.reshape(-1, 1) if p.ndim == 1 else p
            if p.shape[1] == 1:
                return np.hstack([1 - p, p])
            return p
    return _BoosterWrapper(booster)

def load_model(model_uri: Optional[str] = None, model: Optional[Any] = None):
    if model is not None:
        return model
    if not model_uri:
        raise ValueError("Provide model_uri or model object")

    if _is_mlflow_uri(model_uri):
        return _load_mlflow_model(model_uri)

    path = Path(model_uri)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_uri}")

    s = path.suffix.lower()
    if s in {".pkl", ".joblib"}:
        return _load_joblib(str(path))
    if s == ".txt":
        return _load_lightgbm_booster(str(path))

    return _load_joblib(str(path))

def _predict_proba(model, X):
    import numpy as np
    import math
    if hasattr(model, "predict_proba"):
        P = model.predict_proba(X)
        P = np.asarray(P)
        if P.ndim == 2 and P.shape[1] >= 2:
            return P[:, 1]
        if P.ndim == 2 and P.shape[1] == 1:
            return P[:, 0]
        return P
    if hasattr(model, "predict"):
        y = np.asarray(model.predict(X))
        if y.min() >= 0 and y.max() <= 1:
            return y
        return 1.0 / (1.0 + np.exp(-y))
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X))
        return 1.0 / (1.0 + np.exp(-s))
    raise TypeError("Model does not expose predict_proba/predict/decision_function")

def _align_features(X, required_cols: Optional[list]):
    import pandas as pd
    if required_cols is None:
        return X
    X = X.copy()
    for c in required_cols:
        if c not in X.columns:
            X[c] = 0.0
    return X[required_cols]

def filter_with_model(
    *,
    model_uri: Optional[str] = None,
    model: Optional[Any] = None,
    X: Optional["pd.DataFrame"] = None,
    entries: Optional["pd.DataFrame"] = None,
    threshold: float = 0.5,
    feature_names: Optional[list] = None,
) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    if X is None and entries is None:
        raise ValueError("Provide X and/or entries")

    mdl = load_model(model_uri=model_uri, model=model)

    req = None
    if feature_names:
        req = list(feature_names)
    else:
        try:
            req = list(getattr(getattr(mdl, "booster", mdl), "feature_name")())
        except Exception:
            req = None

    if X is not None:
        X2 = _align_features(X, req)
        proba = _predict_proba(mdl, X2)
        proba = np.asarray(proba).reshape(-1)
        proba_s = pd.Series(proba, index=X2.index, name="p1")
        mask = proba_s >= float(threshold)
    else:
        idx = entries.index if entries is not None else None
        proba_s = pd.Series(np.nan, index=idx, name="p1") if idx is not None else None
        mask = pd.Series(True, index=idx, name="keep") if idx is not None else None

    filtered = entries
    if entries is not None and mask is not None:
        common = entries.index.intersection(mask.index)
        filtered = entries.loc[common][mask.loc[common]]

    total = int(len(entries)) if entries is not None else int(len(proba_s)) if proba_s is not None else 0
    kept = int(len(filtered)) if filtered is not None else int(mask.sum()) if mask is not None else 0
    stats = {
        "kept": kept,
        "total": total,
        "accept_rate": (kept / total) if total else 0.0,
        "threshold": float(threshold),
    }

    return {
        "entries": filtered if entries is not None else None,
        "mask": mask,
        "proba": proba_s,
        "threshold": float(threshold),
        "stats": stats,
    }
