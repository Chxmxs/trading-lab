"""
trading_lab: guardrails & reproducibility utilities
"""
from __future__ import annotations
import os, random
from typing import Optional
import numpy as np
import pandas as pd

__version__ = "0.1.0"

# Hard invariants
DATASET_START = pd.Timestamp("2017-01-01T00:00:00Z")
DATASET_END   = pd.Timestamp("2025-03-14T23:59:59Z")
TIMEZONE = "UTC"
GLOBAL_SEED = 777

def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        # Makes Python's hashing deterministic
        import sys, importlib
        if "hashlib" in sys.modules:
            importlib.reload(sys.modules["hashlib"])
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch not installed or no GPU; that's fine
        pass

def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if not isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise TypeError("DataFrame index must be DatetimeIndex/PeriodIndex")
    if getattr(idx, "tz", None) is None:
        df = df.tz_localize(TIMEZONE)
    else:
        df = df.tz_convert(TIMEZONE)
    return df

def clip_to_dataset_window(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_utc_index(df)
    return df[(df.index >= DATASET_START) & (df.index <= DATASET_END)]

def right_closed_resample(df: pd.DataFrame, rule: str, label: str = "right") -> pd.DataFrame:
    # Enforce right-closed bars (close aligned to period end)
    df = ensure_utc_index(df)
    return df.resample(rule, label=label, closed="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })