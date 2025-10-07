# -*- coding: utf-8 -*-
"""
companion.ml.auto_tuner — Stage 13: Adaptive Tuner v2

Learn priors from historical Optuna trials and emit narrowed search spaces
for new studies. Deterministic; logs to MLflow; PS5-safe; no installs.

Run example:
python -m companion.ml.auto_tuner ^
  --history artifacts\optuna\trials.parquet ^
  --strategy SOPRRegimeBand --symbol BTCUSD --timeframe 15m ^
  --policy configs\tuning_policy.json ^
  --out configs\warm_spaces\SOPRRegimeBand_BTCUSD_15m.json
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json, math, os, random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Optional model backends
try:
    import lightgbm as lgb  # type: ignore
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    import xgboost as xgb  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import mlflow  # tradebot env

SEED = 777
np.random.seed(SEED)
random.seed(SEED)

@dataclass
class Policy:
    objective_col: str = "objective"
    top_frac: float = 0.25
    min_trials: int = 60
    half_life_days: float = 90.0
    min_width_ratio: float = 0.15
    expand_margin: float = 0.05
    kfolds: int = 5
    model_type: str = "auto"   # auto|lgb|xgb|gbr
    tag_stage: str = "13"

    @staticmethod
    def load(path: Path | None) -> "Policy":
        # Accepts UTF-8 with or without BOM (PowerShell 5 often writes BOM).
        if path is None or not path.exists():
            return Policy()
        try:
            txt = path.read_text(encoding="utf-8-sig")
            data = json.loads(txt)
        except Exception:
            # Fallback: strip an explicit BOM if present
            raw = path.read_text(encoding="utf-8", errors="replace")
            if raw and raw[0] == "\ufeff":
                raw = raw.lstrip("\ufeff")
            data = json.loads(raw)
        return Policy(**{**Policy().__dict__, **data})


def utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")

def _as_datetime(s):
    if s is None:
        return None
    if isinstance(s, pd.Timestamp):
        return s.to_pydatetime()
    try:
        return pd.to_datetime(s, utc=True).to_pydatetime()
    except Exception:
        return None

def load_history(paths: List[Path]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for p in paths:
        if not p.exists():
            continue
        if p.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(p)
        elif p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        elif p.suffix.lower() in {".json", ".jsonl"}:
            try:
                df = pd.read_json(p, lines=p.suffix.lower()==".jsonl")
            except ValueError:
                df = pd.read_json(p)
        else:
            continue
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)
    if "oos_mar" in df.columns and "objective" not in df.columns:
        df["objective"] = df["oos_mar"]
    if "end_time" in df.columns:
        df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        df["end_time"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for k in ("strategy","symbol","timeframe"):
        if k not in df.columns:
            df[k] = "unknown"
    return df

def filter_key(df: pd.DataFrame, strategy: str, symbol: str, timeframe: str) -> pd.DataFrame:
    m = (df["strategy"].astype(str)==strategy) & (df["symbol"].astype(str)==symbol) & (df["timeframe"].astype(str)==timeframe)
    return df.loc[m].copy()

def time_decay_weights(ts: pd.Series, half_life_days: float) -> np.ndarray:
    now = pd.Timestamp.utcnow()
    age_days = (now - pd.to_datetime(ts, utc=True)).dt.total_seconds()/86400.0
    lam = math.log(2.0) / max(1e-6, half_life_days)
    w = np.exp(-lam * age_days)
    w[np.isnan(w)] = 0.0
    return w

def extract_param_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("param_")]

def build_xy(df: pd.DataFrame, policy: Policy) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Deterministic builder:
      - select param_* only
      - coerce numeric (errors->NaN), replace ±inf->NaN
      - drop rows with non-finite target
      - drop all-NaN columns
      - median-impute remaining NaNs
      - drop zero-variance columns
    """
    # Target (objective)
    y = pd.to_numeric(df[policy.objective_col], errors="coerce")
    mask_y = np.isfinite(y.to_numpy(dtype=float, copy=False))
    if not mask_y.any():
        raise ValueError("[AUTO-TUNER] All target values are non-finite for this key.")
    df = df.loc[mask_y].copy()
    y = y.loc[mask_y].astype(float)

    # Params
    params = extract_param_cols(df)
    if not params:
        raise ValueError("[AUTO-TUNER] No param_* columns found for this (strategy, symbol, timeframe).")
    X = df[params].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Drop all-NaN columns
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        X = X.drop(columns=all_nan)
        if X.shape[1] == 0:
            raise ValueError("[AUTO-TUNER] All param_* columns are NaN for this key.")

    # Median-impute remaining NaNs
    if X.isna().any().any():
        med = X.median(numeric_only=True)
        X = X.fillna(med)
        if X.isna().any().any():
            X = X.fillna(0.0)

    # Drop zero-variance columns
    zero_var = X.nunique(dropna=False)
    zero_var = zero_var[zero_var <= 1].index.tolist()
    if zero_var:
        X = X.drop(columns=zero_var)
        if X.shape[1] == 0:
            raise ValueError("[AUTO-TUNER] No usable param_* columns remain after cleaning (all zero-variance).")

    # Weights for remaining rows
    w = time_decay_weights(df.get("end_time", pd.Series([pd.Timestamp.utcnow()] * len(df))), policy.half_life_days)
    w = np.asarray(w, dtype=float)
    w[~np.isfinite(w)] = 0.0

    return X.astype(float), y.astype(float), w




def train_model(X: pd.DataFrame, y: pd.Series, w: np.ndarray, policy: Policy) -> Tuple[object, Dict[str, float]]:
    """
    Train meta-model using NumPy arrays (positional CV).
    Prefer LightGBM/XGBoost (NaN-tolerant); fallback to GBR.
    Always returns (model, summary).
    """
    # Choose backend
    if policy.model_type == "auto":
        model_type = "lgb" if _HAS_LGB else ("xgb" if _HAS_XGB else "gbr")
    else:
        model_type = policy.model_type

    # NumPy views + final finiteness guard
    Xv = X.to_numpy(dtype=float, copy=False)
    yv = y.to_numpy(dtype=float, copy=False)
    wv = np.asarray(w, dtype=float)

    Xv[~np.isfinite(Xv)] = 0.0
    yv[~np.isfinite(yv)] = np.nan
    keep = np.isfinite(yv)
    if keep.sum() < 2:
        raise ValueError("[AUTO-TUNER] Not enough finite target rows after cleaning.")
    Xv, yv, wv = Xv[keep], yv[keep], wv[keep]

    n_rows = Xv.shape[0]
    splits = max(2, min(policy.kfolds, n_rows))
    kf = KFold(n_splits=splits, shuffle=True, random_state=SEED)

    def _fit_new():
        if model_type == "lgb":
            return lgb.LGBMRegressor(random_state=SEED, n_estimators=400, learning_rate=0.05, num_leaves=31)
        if model_type == "xgb":
            return xgb.XGBRegressor(random_state=SEED, n_estimators=400, learning_rate=0.05,
                                    max_depth=6, subsample=0.9, colsample_bytree=0.9)
        return GradientBoostingRegressor(random_state=SEED)

    scores: List[float] = []
    for tr, te in kf.split(Xv):
        m = _fit_new()
        try:
            m.fit(Xv[tr], yv[tr], sample_weight=wv[tr])
        except TypeError:
            m.fit(Xv[tr], yv[tr])
        yhat = m.predict(Xv[te])
        scores.append(r2_score(yv[te], yhat))

    model = _fit_new()
    try:
        model.fit(Xv, yv, sample_weight=wv)
    except TypeError:
        model.fit(Xv, yv)

    return model, {
        "cv_r2_mean": float(np.mean(scores)),
        "cv_r2_std": float(np.std(scores)),
        "model_type": model_type,
        "rows": int(n_rows),
        "features": int(Xv.shape[1]),
        "splits": int(splits)
    }



def derive_warm_space(df_key: pd.DataFrame, policy: Policy) -> Dict[str, dict]:
    """
    Compute narrowed per-parameter ranges from the TOP fraction of trials,
    with time-decay weights and safety margins. Self-contained (no external helpers).
    """
    if df_key.empty:
        return {}

    if policy.objective_col not in df_key.columns:
        raise ValueError(f"[AUTO-TUNER] Missing objective column '{policy.objective_col}'.")

    params = extract_param_cols(df_key)
    if not params:
        return {}

    # --- select TOP fraction by objective ---
    dfx = df_key.copy()
    dfx[policy.objective_col] = pd.to_numeric(dfx[policy.objective_col], errors="coerce")
    dfx = dfx[np.isfinite(dfx[policy.objective_col].to_numpy(dtype=float, copy=False))]
    if dfx.empty:
        raise ValueError("[AUTO-TUNER] No finite objective rows to derive warm space.")

    dfx = dfx.sort_values(policy.objective_col, ascending=False, kind="mergesort").reset_index(drop=True)
    k = max(1, int(len(dfx) * max(0.0, min(1.0, policy.top_frac))))
    top_df = dfx.iloc[:k].copy()

    # --- weights (newer trials weigh more) ---
    def _tw(ts: pd.Series) -> np.ndarray:
        now = pd.Timestamp.utcnow()
        age_days = (now - pd.to_datetime(ts, utc=True, errors="coerce")).dt.total_seconds() / 86400.0
        lam = math.log(2.0) / max(1e-6, policy.half_life_days)
        w = np.exp(-lam * age_days)
        w[~np.isfinite(w)] = 0.0
        return w.to_numpy(dtype=float, copy=False)

    w_top = _tw(top_df.get("end_time", pd.Series([pd.Timestamp.utcnow()] * len(top_df))))
    out: Dict[str, dict] = {}

    for p in params:
        # coerce to numeric for both full and top slices
        v_all = pd.to_numeric(dfx[p], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        v_top = pd.to_numeric(top_df[p], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)

        # guard: if all NaN for this param, skip
        if not np.isfinite(v_all).any():
            continue

        # Weighted 10/50/90 from TOP slice (fallback to unweighted if all weights zero)
        mask_top = np.isfinite(v_top)
        v_top_f = v_top[mask_top]
        w_top_f = w_top[mask_top] if mask_top.any() else np.array([], dtype=float)

        if v_top_f.size == 0:
            # no finite in top → fallback to full finite values, unweighted
            v_top_f = v_all[np.isfinite(v_all)]
            w_top_f = np.ones_like(v_top_f, dtype=float)

        def _wq(vals: np.ndarray, w: np.ndarray, qs: List[float]) -> List[float]:
            # stable weighted quantile
            if vals.size == 0:
                return [float("nan") for _ in qs]
            order = np.argsort(vals)
            vv = vals[order]
            ww = np.asarray(w, dtype=float)
            ww = ww[order] if ww.size == vals.size else np.ones_like(vv, dtype=float)
            ww[~np.isfinite(ww)] = 0.0
            cw = np.cumsum(ww)
            tot = float(cw[-1]) if cw.size else 0.0
            if tot <= 0.0:
                cw = np.arange(1, vv.size + 1, dtype=float)
                tot = cw[-1]
            res = []
            for q in qs:
                t = float(q) * tot
                idx = int(np.searchsorted(cw, t, side="left"))
                idx = max(0, min(idx, vv.size - 1))
                res.append(float(vv[idx]))
            return res

        q10, q50, q90 = _wq(v_top_f, w_top_f, [0.10, 0.50, 0.90])

        # full-span guards
        v_all_f = v_all[np.isfinite(v_all)]
        gmin = float(np.nanmin(v_all_f))
        gmax = float(np.nanmax(v_all_f))
        span = max(1e-12, gmax - gmin)

        new_low = max(gmin, (q10 if np.isfinite(q10) else gmin) - policy.expand_margin * span)
        new_high = min(gmax, (q90 if np.isfinite(q90) else gmax) + policy.expand_margin * span)

        # minimum width ratio of original span
        min_width = policy.min_width_ratio * span
        if (new_high - new_low) < min_width:
            c = 0.5 * (new_high + new_low)
            new_low = max(gmin, c - 0.5 * min_width)
            new_high = min(gmax, c + 0.5 * min_width)

        # integer vs float suggestion
        # (if all observed finite values for p are integer-like, treat as int)
        finite_vals = v_all_f
        is_int_like = np.all(np.isfinite(finite_vals)) and np.allclose(finite_vals, np.round(finite_vals))
        if is_int_like:
            new_low = math.floor(new_low)
            new_high = math.ceil(new_high)
            suggest = "int"
        else:
            suggest = "float"

        out[p] = {
            "suggest": suggest,
            "low": float(new_low),
            "high": float(new_high),
            "median": float(q50) if np.isfinite(q50) else float(np.nan),
            "global_min": gmin,
            "global_max": gmax
        }

    return out


def log_to_mlflow(artifacts: Dict[str, str], meta: Dict[str, object], policy: Policy) -> None:
    """
    Robust MLflow logger:
      - sets tracking URI to local file store
      - ensures an experiment exists (creates if missing)
      - logs params/metrics/artifacts under artifact_path="auto_tuner_v2"
      - never raises (prints warning and returns on failure)
    """
    try:
        import mlflow
        from mlflow.exceptions import MlflowException

        mlflow.set_tracking_uri("file:./mlruns")

        # Ensure an experiment exists (create if needed)
        exp_name = "trading-lab"
        try:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp is None:
                mlflow.create_experiment(exp_name)
        except Exception:
            # final fallback: ignore experiment handling; MLflow will use default if present
            pass

        # Use the named experiment (creates if missing)
        try:
            mlflow.set_experiment(exp_name)
        except Exception:
            # fallback to whatever default exists
            pass

        with mlflow.start_run(run_name=f"auto_tuner_v2_{utcnow_str()}"):
            mlflow.set_tag("stage", policy.tag_stage)
            mlflow.set_tag("module", "auto_tuner_v2")

            # Log params (simple scalars only)
            for k, v in meta.items():
                if isinstance(v, (int, float, str, bool)):
                    try:
                        mlflow.log_param(k, v)
                    except Exception:
                        pass

            # Key metrics
            for k in ("cv_r2_mean", "cv_r2_std"):
                if k in meta and isinstance(meta[k], (int, float)):
                    try:
                        mlflow.log_metric(k, float(meta[k]))
                    except Exception:
                        pass

            # Artifacts
            for name, path in artifacts.items():
                p = Path(path)
                if p.exists():
                    try:
                        mlflow.log_artifact(str(p), artifact_path="auto_tuner_v2")
                    except Exception:
                        pass

    except Exception as e:
        print(f"[AUTO-TUNER] MLflow logging skipped: {e}")


def run(history_paths: List[Path], strategy: str, symbol: str, timeframe: str,
        policy_path: Path | None, out_path: Path) -> int:
    policy = Policy.load(policy_path)

    df_all = load_history(history_paths)
    if df_all.empty:
        print("[AUTO-TUNER] No history rows found — nothing to do.")
        return 2

    df_key = filter_key(df_all, strategy, symbol, timeframe)
    if len(df_key) < policy.min_trials:
        print(f"[AUTO-TUNER] Not enough trials for {strategy}/{symbol}/{timeframe}: "
              f"{len(df_key)} < {policy.min_trials}")
        return 3

    # Build matrices
    try:
        X, y, w = build_xy(df_key, policy)
    except Exception as e:
        print(f"[AUTO-TUNER] BuildXY failed: {e}")
        return 4

    # Train meta-model
    try:
        model, summary = train_model(X, y, w, policy)
    except Exception as e:
        print(f"[AUTO-TUNER] Training failed: {e}")
        return 5

    # Derive warm search space
    try:
        warm = derive_warm_space(df_key, policy)
    except Exception as e:
        print(f"[AUTO-TUNER] Warm-space derivation failed: {e}")
        return 6

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "generated_at": utcnow_str(),
        "policy": policy.__dict__,
        "model": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in summary.items()},
        "warm_space": warm
    }, indent=2), encoding="utf-8")

    # Feature importances (best-effort)
    fi_path = out_path.parent / (out_path.stem + "_feature_importance.csv")
    try:
        if hasattr(model, "feature_importances_"):
            pd.DataFrame({
                "feature": list(X.columns),
                "importance": list(getattr(model, "feature_importances_"))
            }).sort_values("importance", ascending=False).to_csv(fi_path, index=False)
    except Exception:
        pass

    # MLflow logging
    artifacts = {"warm_space": str(out_path)}
    if fi_path.exists():
        artifacts["feature_importance"] = str(fi_path)
    meta = {
        "strategy": strategy, "symbol": symbol, "timeframe": timeframe,
        "rows_total": int(len(df_all)), "rows_key": int(len(df_key)), **summary
    }
    log_to_mlflow(artifacts, meta, policy)

    print(f"[AUTO-TUNER] Wrote warm space → {out_path}")
    return 0


def _parse_args(argv: List[str] | None = None):
    import argparse
    p = argparse.ArgumentParser(description="Adaptive Tuner v2 — warm space generator")
    p.add_argument("--history", nargs="+", required=True, help="One or more parquet/CSV/JSON(L) trial-history files")
    p.add_argument("--strategy", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--timeframe", required=True)
    p.add_argument("--policy", default="configs/tuning_policy.json")
    p.add_argument("--out", required=True, help="Path to write warm space JSON")
    return p.parse_args(argv)

def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)
    history_paths = [Path(h) for h in args.history]
    out_path = Path(args.out)
    policy_path = Path(args.policy) if args.policy else None
    return run(history_paths, args.strategy, args.symbol, args.timeframe, policy_path, out_path)

if __name__ == "__main__":
    raise SystemExit(main())
