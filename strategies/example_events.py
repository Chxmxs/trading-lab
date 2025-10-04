from __future__ import annotations
import pandas as pd
from typing import Optional, Dict, Any
from .base import StrategyBase

class EventsOnlyStrategy(StrategyBase):
    """
    Minimal strategy for CV:
      - Implements build_events(df) -> DataFrame with UTC index and 't1' (UTC) end time.
      - No trading logic; base.run() will return flat equity + empty trades.
    """

    def build_events(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 2:
            # Need at least 2 timestamps to form [start, t1)
            raise ValueError("Need at least 2 rows in df to build event windows.")
        # Ensure UTC index
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        # Create one event per bar using next bar as t1 (closed-open [t, t1))
        starts = idx[:-1]
        ends = idx[1:]
        events = pd.DataFrame(index=starts)
        events["t1"] = ends
        # (Optional) place to attach labels/features later:
        # events["label"] = 0
        # events["meta_features"] = None
        return events

__all__ = ["EventsOnlyStrategy"]
