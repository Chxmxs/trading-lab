from __future__ import annotations

import pandas as pd
from typing import Mapping, Any

from strategies.base import StrategyBase

class MyStrategy(StrategyBase):
    """
    Phase 4 subclass scaffold.

    Implement your actual rules later by overriding these hooks.
    For now, this is a template that shows where to put your code.
    """

    def generate_signals(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame | None:
        # Return a Series or DataFrame indexed by UTC timestamps.
        # Values should be in {-1, 0, +1}. For now, no trades:
        return None

    def position_sizing(self, bar: Any, account: Mapping[str, float]) -> float | None:
        # Example: 10% of equity position. Keep None for default sizing for now.
        # return 0.10
        return None

    def risk_stops(self, bar: Any, position: Any) -> dict | None:
        # Example (uncomment when you add entries):
        # return {"stop_loss_pct": 0.02, "take_profit_pct": 0.04, "hold_bars": 3}
        return None

    def session_filter(self, ts: pd.Timestamp) -> bool:
        # Return False to skip a bar at timestamp ts. Default: keep all.
        return True

    def on_start(self, context: Mapping[str, Any]) -> None:
        # Optional lifecycle hook
        return None

    def on_bar(self, context: Mapping[str, Any]) -> None:
        # Optional lifecycle hook
        return None

    def on_finish(self, context: Mapping[str, Any]) -> None:
        # Optional lifecycle hook
        return None
