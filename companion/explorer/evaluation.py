"""
evaluation.py
-------------
Cross-validation and metrics engine for Trading-Lab Explorer.
Supports mlfinlab PurgedKFold, UTC auto-patching, retry/backoff, and MLflow logging.
"""


from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from companion.patch_registry import apply_all_patches
from companion.error_handling import capture_errors, retry_with_backoff
import json

import numpy as np
import pandas as pd

# Optional MLflow support
try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False


# --- Auto-patching & normalization helper ---
def _autopatch_and_normalize_df(df):
    """
    Applies all registered patches and enforces a UTC DatetimeIndex.
    Returns the patched DataFrame. Never raises; logs internally.
    """
    try:
        df = apply_all_patches(df)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Auto-patching failed (ignored).")

    # Enforce UTC DatetimeIndex if possible
    try:
        if hasattr(df, "index") and not isinstance(df.index, type(None)):
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
    except Exception:
        logging.getLogger(__name__).exception("Failed to enforce UTC index (ignored).")

    return df
# --------------------------------------------


# --- BEGIN: mlfinlab PurgedKFold imports and flags (multi-path support) ---
_HAS_MLFINLAB = False
PurgedKFold = None  # type: ignore

try:
    # Newer/common path
    from mlfinlab.cross_validation import PurgedKFold as _PKF  # type: ignore
    PurgedKFold = _PKF
    _HAS_MLFINLAB = True
except Exception:
    try:
        # Older releases keep it deeper:
        from mlfinlab.cross_validation.cross_validation import PurgedKFold as _PKF  # type: ignore
        PurgedKFold = _PKF
        _HAS_MLFINLAB = True
    except Exception:
        try:
            # Some builds expose via model_selection
            from mlfinlab.model_selection import PurgedKFold as _PKF  # type: ignore
            PurgedKFold = _PKF
            _HAS_MLFINLAB = True
        except Exception:
            _HAS_MLFINLAB = False
# --- END: mlfinlab PurgedKFold imports and flags ---

logger = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------
@dataclass
class CVConfig:
    n_splits: int = 5
    embargo_pct: float = 0.0  # fraction of rows used as embargo (0.01 -> 1% of dataset length)
    purge_length: Optional[int] = None  # kept for compatibility; not all PurgedKFold variants use it


# -----------------------------
# Helpers
# -----------------------------
def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures df.index is a UTC DatetimeIndex for PurgedKFold.
    If tz-naive, localize to UTC; if tz-aware, convert to UTC.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas.DatetimeIndex for PurgedKFold.")
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    else:
        df = df.tz_convert("UTC")
    return df


def compute_performance_metrics(equity: pd.Series, trades: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Compute basic metrics on a test equity curve.
    Expected: equity is a monotonic-time series of equity values (price-level, not returns).
    """
    metrics: Dict[str, float] = {}

    if equity is None or len(equity) < 2:
        logger.warning("Equity series too short to compute metrics.")
        return {}

    equity = equity.sort_index()
    returns = equity.pct_change().dropna()

    # Time span (years)
    t0, t1 = equity.index[0], equity.index[-1]
    delta_years = (t1 - t0).total_seconds() / (365.25 * 24 * 3600)
    if delta_years <= 0:
        logger.warning("Non-positive time span for equity curve.")
        return {}

    cumulative_return = float(equity.iloc[-1]) / float(equity.iloc[0])
    cagr = cumulative_return ** (1.0 / delta_years) - 1.0

    running_max = equity.cummax()
    drawdowns = equity / running_max - 1.0
    max_dd = float(drawdowns.min())

    # MAR ratio
    mar = (cagr / abs(max_dd)) if max_dd != 0 else float("nan")

    # Simple daily Sharpe approximation (assumes equity sampled daily-ish)
    ret_std = float(returns.std(ddof=0)) if len(returns) else 0.0
    sharpe = float("nan") if ret_std == 0 else float(returns.mean() / ret_std) * math.sqrt(252)

    metrics.update({
        "cagr": float(cagr),
        "max_drawdown": float(max_dd),
        "mar": float(mar),
        "sharpe": float(sharpe),
        "num_trades": float(len(trades) if trades is not None else 0),
    })
    return metrics


def _simple_sequential_splits(n_obs: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Very simple sequential splits: folds are contiguous chunks of the time series.
    """
    indices: List[Tuple[np.ndarray, np.ndarray]] = []
    if n_splits <= 1:
        return indices
    fold_size = n_obs // n_splits
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n_obs
        train_idx = np.arange(0, test_start, dtype=int)
        test_idx = np.arange(test_start, test_end, dtype=int)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        indices.append((train_idx, test_idx))
    return indices


def _kfold_fallback_splits(n_obs: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Fallback to scikit-learn KFold when mlfinlab isn't available or fails.
    """
    try:
        from sklearn.model_selection import KFold  # type: ignore
        kf = KFold(n_splits=n_splits, shuffle=False)
        return list(kf.split(np.arange(n_obs, dtype=int)))
    except Exception as e:
        logger.warning("KFold fallback unavailable (%s). Using simple sequential splits.", e)
        return _simple_sequential_splits(n_obs, n_splits)


def _purged_kfold_splits(data: pd.DataFrame, cv_config: CVConfig) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Try to build PurgedKFold splits with mlfinlab. If anything goes wrong, return None.
    Supports both 'embargo_td' (int rows) and 'pct_embargo' (float fraction) ctor styles.
    """
    if not _HAS_MLFINLAB or PurgedKFold is None:
        return None

    try:
        data = _ensure_utc_index(data)
        n_splits = int(getattr(cv_config, "n_splits", 5))
        emb_frac = float(getattr(cv_config, "embargo_pct", 0.0))
        emb_rows = int(max(0, round(emb_frac * len(data))))

        split_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

        # Try constructor with embargo_td (rows)
        try:
            pkf = PurgedKFold(n_splits=n_splits, samples_info_sets=data.index, embargo_td=emb_rows)  # type: ignore
            split_indices = list(pkf.split(None))
        except Exception:
            split_indices = None

        # If that failed, try pct_embargo (fraction)
        if not split_indices:
            try:
                pkf = PurgedKFold(n_splits=n_splits, samples_info_sets=data.index, pct_embargo=emb_frac)  # type: ignore
                split_indices = list(pkf.split(None))
            except Exception:
                split_indices = None

        return split_indices or None
    except Exception as e:
        logger.warning("PurgedKFold unavailable or incompatible (%s). Falling back.", e)
        return None



# -----------------------------
# Main CV evaluator
# -----------------------------
@capture_errors(run_name="evaluate_strategy_cv")
@retry_with_backoff(max_retries=2, base_delay=3)
def evaluate_strategy_cv(
    strategy_func: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, Any]],
    data: pd.DataFrame,
    run_config: Dict[str, Any],
    cv_config: CVConfig,
    timestamp_column: str = "timestamp",
) -> Dict[str, float]:
    """
    Cross-validate a strategy function over time-series data.

    Parameters
    ----------
    strategy_func : Callable
        Function that takes (dataframe, run_config) and returns a dict:
            {
                "equity": pd.Series,               # equity curve indexed by time
                "trades": Optional[pd.DataFrame]   # optional trade ledger
            }
        The evaluator uses ONLY the test fold results to compute metrics.

    data : pd.DataFrame
        Time-series market data. Must be or be convertible to a UTC DatetimeIndex.
        If index is not DatetimeIndex, provide a timestamp column (default "timestamp"),
        or a "date" column (ISO8601), or a numeric "timestamp" in seconds since epoch.

    run_config : Dict[str, Any]
        Any runtime parameters for the strategy function.

    cv_config : CVConfig
        Cross-validation configuration (n_splits, embargo_pct).

    timestamp_column : str
        Column to interpret as timestamp if the index is not a DatetimeIndex.
    """
    logger.info(
        "Starting cross-validation for strategy_func=%s | n_splits=%d",
        getattr(strategy_func, "__name__", str(strategy_func)),
        getattr(cv_config, "n_splits", 5),
    )


    # --- Ensure datetime index (UTC) ---
    if not isinstance(data.index, pd.DatetimeIndex):
        # Try provided column name or common alternatives
        if timestamp_column in data.columns:
            ts = data[timestamp_column]
            if pd.api.types.is_numeric_dtype(ts):
                data["__ts__"] = pd.to_datetime(ts.astype("int64"), unit="s", utc=True)
            else:
                data["__ts__"] = pd.to_datetime(ts, utc=True)
        elif "date" in data.columns:
            data["__ts__"] = pd.to_datetime(data["date"], utc=True)
        elif "timestamp" in data.columns:
            ts = data["timestamp"]
            if pd.api.types.is_numeric_dtype(ts):
                data["__ts__"] = pd.to_datetime(ts.astype("int64"), unit="s", utc=True)
            else:
                data["__ts__"] = pd.to_datetime(ts, utc=True)
        else:
            raise ValueError(
                f"Data must contain a '{timestamp_column}' or 'date'/'timestamp' column "
                "when the index is not a DatetimeIndex."
            )

        data = data.set_index(data["__ts__"]).drop(columns=["__ts__"])
    # Ensure UTC
    data = _ensure_utc_index(data)

    # --- Basic sanity for CV ---
    n_obs = int(len(data))
    if n_obs < int(cv_config.n_splits) + 1:
        raise ValueError("Not enough observations for the requested number of splits.")

    # --- Build split indices (PurgedKFold -> KFold -> simple) ---
    split_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = _purged_kfold_splits(data, cv_config)
    if split_indices is None:
        split_indices = _kfold_fallback_splits(n_obs, int(cv_config.n_splits))
    if not split_indices:
        logger.warning("No valid CV splits could be created.")
        return {}

    # --- Evaluate each fold (use ONLY test fold for metric computation) ---
    fold_metrics: List[Dict[str, float]] = []
    for fold_num, (train_idx, test_idx) in enumerate(split_indices):
        try:
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            if train_data.empty or test_data.empty:
                continue

            # Strategy function is expected to be pure on the provided data subset.
            # You can adapt this to fit-on-train / eval-on-test if your strategy supports it.
            _ = strategy_func(train_data, run_config)  # optional: fit/cache/prewarm

            test_result = strategy_func(test_data, run_config)
            if not isinstance(test_result, dict) or "equity" not in test_result:
                logger.warning("strategy_func did not return expected dict with 'equity'. Skipping fold %d.", fold_num)
                continue

            equity = test_result["equity"]
            trades = test_result.get("trades")
            if not isinstance(equity, pd.Series) or len(equity) < 2:
                logger.warning("Invalid or too-short equity in fold %d.", fold_num)
                continue

            metrics = compute_performance_metrics(equity, trades)
            if metrics:
                fold_metrics.append(metrics)
        except Exception as exc:
            logger.exception("Error evaluating strategy on fold %d: %s", fold_num, exc)
            continue

    if not fold_metrics:
        return {}

    # --- Aggregate across folds ---
    aggregated: Dict[str, float] = {}
    all_keys = set().union(*[m.keys() for m in fold_metrics])
    for k in all_keys:
        vals = [float(m[k]) for m in fold_metrics if (k in m and not (isinstance(m[k], float) and math.isnan(m[k])))]
        if not vals:
            aggregated[k] = float("nan")
        elif k == "num_trades":
            aggregated[k] = float(sum(vals))
        else:
            aggregated[k] = float(np.mean(vals))

    # Optional: include n_folds actually used
    aggregated["folds"] = float(len(fold_metrics))

    # --- Optional MLflow logging ---
    if _HAS_MLFLOW:
        try:
            with mlflow.start_run(run_name=str(getattr(strategy_func, "__name__", "strategy"))):
                for key, val in aggregated.items():
                    if isinstance(val, (int, float)) and not math.isnan(val):
                        mlflow.log_metric(key, float(val))
                mlflow.log_params(run_config or {})
                mlflow.log_text(json.dumps(aggregated, indent=2), "cv_metrics.json")
        except Exception as e:
            logger.warning("MLflow logging skipped: %s", e)
    logger.info("Completed CV for %s | folds=%d | metrics=%s",
                getattr(strategy_func, "__name__", str(strategy_func)),
                int(aggregated.get("folds", 0)),
                {k: round(v, 4) for k, v in aggregated.items()})


    return aggregated
