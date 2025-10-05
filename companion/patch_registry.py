import logging
from typing import Callable, List
import pandas as pd

log = logging.getLogger("companion.patch_registry")

# Global registry of patchers (callables df->df)
_PATCHERS: List[Callable[[pd.DataFrame], pd.DataFrame]] = []

def register_patcher(func: Callable[[pd.DataFrame], pd.DataFrame]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Decorator to register a patcher in global order."""
    _PATCHERS.append(func)
    log.info("Registered patcher: %s", func.__name__)
    return func

def apply_all_patches(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all registered patchers in order. Return the patched DataFrame."""
    out = df.copy()
    for func in list(_PATCHERS):
        try:
            out = func(out)
        except Exception as e:
            log.error("apply_all_patches: %s failed -> %s", func.__name__, e)
    return out

@register_patcher
def ohlc_patch(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected OHLC columns casing and presence; pass-through if already OK."""
    out = df.copy()
    # Normalize common alias casing (no truthiness on DataFrame!)
    cols = {c.lower(): c for c in out.columns}
    # If columns already include open/close, do nothing
    expected = ["open", "close"]
    if all(name in cols for name in expected):
        return out
    # Sometimes files use 'Open','Close' capitalization — just rename
    rename_map = {}
    if "open" not in cols and "Open" in out.columns:
        rename_map["Open"] = "open"
    if "close" not in cols and "Close" in out.columns:
        rename_map["Close"] = "close"
    if rename_map:
        out = out.rename(columns=rename_map)
    return out

@register_patcher
def timestamp_patch(df: pd.DataFrame) -> pd.DataFrame:
    """Create UTC DatetimeIndex from common timestamp columns; drop the column after."""
    out = df.copy()

    timestamp_col = None
    for cand in ["timestamp", "Timestamp", "date", "Date", "datetime", "Datetime"]:
        if cand in out.columns:
            timestamp_col = cand
            break

    if timestamp_col is None:
        # nothing to do
        return out

    ts = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
    out = out.drop(columns=[timestamp_col])
    out.index = ts
    out.index.name = "timestamp"
    return out
