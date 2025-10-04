from __future__ import annotations
from typing import Dict, List, Optional, TypedDict, Any
import pandas as pd

TRADE_COLUMNS: List[str] = [
    "symbol","entry_time","exit_time","side","qty",
    "entry_price","exit_price","fees","slippage",
    "pnl","pnl_pct","mae","mfe","bars_held",
    "entry_reason","exit_reason","stop_triggered"
]

class EntriesDF(TypedDict, total=False):
    """
    A minimal typed container description for ML hooks.

    Expected index:
        - UTC pandas.DatetimeIndex (tz-aware, UTC)
    Accepted columns (all optional):
        - 'is_candidate': bool/int; True/1 marks a candidate entry event
        - 'meta_features': Any; bag for user features (array/struct/etc.)
        - 'label': int/float; optional supervised label
        - 'prob': float; optional probability from classifier

    Notes:
        - Hooks also accept a pd.Series of booleans/ints indexed by UTC timestamps.
        - Implementations should treat missing 'is_candidate' as True for all rows.
    """
    is_candidate: Any
    meta_features: Any
    label: Any
    prob: Any

class RunConfig(TypedDict, total=False):
    symbol: str
    timeframe: str
    cash: float
    fees: Optional[float]
    slippage: Optional[float]
    start: Optional[str]  # ISO UTC
    end: Optional[str]    # ISO UTC
    params: Dict[str, Any]

class RunResult(TypedDict, total=False):
    equity: pd.Series   # UTC index, name='equity'
    trades: pd.DataFrame  # exact schema TRADE_COLUMNS
    stats: Dict[str, Any]
