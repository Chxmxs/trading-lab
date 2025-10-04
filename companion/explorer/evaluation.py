from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class CVConfig:
    n_splits: int = 5
    embargo_pct: float = 0.0
    purge_length: Optional[int] = None

def compute_performance_metrics(equity: pd.Series, trades: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    equity = equity.sort_index()
    returns = equity.pct_change().dropna()
    if len(equity) < 2:
        logger.warning("Equity series too short to compute metrics.")
        return {}
    t0, t1 = equity.index[0], equity.index[-1]
    delta_years = (t1 - t0).total_seconds() / (365.25 * 24 * 3600)
    if delta_years <= 0:
        logger.warning("Non-positive time span for equity curve.")
        return {}
    cumulative_return = equity.iloc[-1] / equity.iloc[0]
    cagr = cumulative_return ** (1.0 / delta_years) - 1
    running_max = equity.cummax()
    drawdowns = equity / running_max - 1
    max_dd = drawdowns.min()
    mar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    sharpe = np.nan if returns.std(ddof=0) == 0 else (returns.mean() / returns.std(ddof=0)) * math.sqrt(252)
    metrics.update({
        "cagr": float(cagr),
        "max_drawdown": float(max_dd),
        "mar": float(mar),
        "sharpe": float(sharpe),
        "num_trades": float(len(trades) if trades is not None else 0),
    })
    return metrics

def simple_split_indices(n_obs: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    indices: List[Tuple[np.ndarray, np.ndarray]] = []
    fold_size = n_obs // n_splits
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n_obs
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        indices.append((train_idx, test_idx))
    return indices

def evaluate_strategy_cv(
    strategy_func: Callable,
    data: pd.DataFrame,
    run_config: Dict[str, Any],
    cv_config: CVConfig,
    timestamp_column: str = "timestamp",
) -> Dict[str, float]:
    # Ensure the data has a datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        # If the expected timestamp column is missing, try alternatives
        if timestamp_column not in data.columns:
            if "date" in data.columns:
                # Use the 'date' column (assumed to be ISO strings)
                data["timestamp"] = pd.to_datetime(data["date"], utc=True)
                timestamp_column = "timestamp"
            elif "timestamp" in data.columns and pd.api.types.is_numeric_dtype(data["timestamp"]):
                # Convert Unix epoch seconds to datetime
                data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s", utc=True)
            else:
                raise ValueError(f"Data must contain '{timestamp_column}' or 'date' column for timestamp.")
        # Set the index to the timestamp column
        data = data.set_index(pd.to_datetime(data[timestamp_column], utc=True))
    else:
        # If it already has a DatetimeIndex, ensure it's UTC
        if data.index.tz is None:
            data = data.tz_localize("UTC")
        else:
            data = data.tz_convert("UTC")

    n_obs = len(data)
    if n_obs < cv_config.n_splits + 1:
        raise ValueError("Not enough observations for the requested number of splits.")

    # Use simple sequential splits; mlfinlab fallback omitted for brevity
    split_indices = simple_split_indices(n_obs, cv_config.n_splits)
    fold_metrics: List[Dict[str, float]] = []
    for fold_num, (train_idx, test_idx) in enumerate(split_indices):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        if train_data.empty or test_data.empty:
            continue
        try:
            train_result = strategy_func(train_data, run_config)
            test_result = strategy_func(test_data, run_config)
            equity = test_result["equity"]
            trades = test_result.get("trades")
            metrics = compute_performance_metrics(equity, trades)
            fold_metrics.append(metrics)
        except Exception as exc:
            logger.exception("Error evaluating strategy on fold %d: %s", fold_num, exc)
            continue
    if not fold_metrics:
        return {}
    aggregated: Dict[str, float] = {}
    keys = fold_metrics[0].keys()
    for k in keys:
        vals = [m[k] for m in fold_metrics if not math.isnan(m[k])]
        if not vals:
            aggregated[k] = float("nan")
        elif k == "num_trades":
            aggregated[k] = float(sum(vals))
        else:
            aggregated[k] = float(np.mean(vals))
    return aggregated
