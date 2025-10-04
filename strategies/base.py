from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional, Union, Tuple, Iterable
import pandas as pd
from pandas import Timestamp
from .types import TRADE_COLUMNS, RunConfig, RunResult

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

def _make_file_logger() -> logging.Logger:
    """Create a per-run file logger (module-level)."""
    logger = logging.getLogger("trading_lab")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fh = logging.FileHandler(os.path.join(_LOG_DIR, f"run_{ts}.log"), encoding="utf-8")
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(name)s :: %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Also echo to console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # Ensure times appear as UTC with Z
    logging.Formatter.converter = lambda *args: pd.Timestamp.utcnow().timetuple()
    return logger

def _ensure_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("Index must be a pandas.DatetimeIndex")
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return pd.DatetimeIndex(idx, tz="UTC")

def _empty_trades_df() -> pd.DataFrame:
    return pd.DataFrame(columns=TRADE_COLUMNS)

def _flat_equity(index: pd.DatetimeIndex, cash: float) -> pd.Series:
    s = pd.Series([cash] * len(index), index=index)
    s.index = _ensure_utc_index(s.index)
    s.name = "equity"
    return s

class StrategyBase:
    """
    StrategyBase with optional ML hooks for MLfinLab-style workflows.
    Compatibility rules:
      - If only rule-based generate_signals(...) is used, ML hooks are ignored.
      - If ML hooks are used and produce no entries after filtering, we log it and
        return a valid no-trades run (equity series + empty trades schema).

    Required outputs (for orchestrator/phase-3 compatibility):
      - equity: pd.Series with UTC index, name='equity'
      - trades: pd.DataFrame with exact columns TRADE_COLUMNS
    """
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, **kwargs):
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.params = kwargs or {}
        self.logger = _make_file_logger().getChild(f"Strategy.{self.name}")

    # -------- Rule-based path (legacy) --------
    def generate_signals(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Optional: Rule-based signals (boolean Series, UTC index) for entries.
        Default None. Subclasses may override.
        """
        return None

    # -------- ML hooks (Phase 3.5) --------
    def propose_entries(self, df: pd.DataFrame) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """
        Optional: Emit candidate entry events.
        Accepted shapes:
          - pd.Series (bool/int), UTC index -> True/1 marks candidate
          - pd.DataFrame, UTC index, optional cols: ['is_candidate','meta_features','label','prob']
        Default: None (no ML usage).
        """
        return None

    def filter_entries(self, entries: Union[pd.Series, pd.DataFrame, None],
                       meta_probs: Optional[Union[pd.Series, Dict[str, float]]] = None
                       ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """
        Optional: Meta-label filtering. Receives output from propose_entries and
        optional meta probabilities. Returns filtered entries; may return empty.
        Default: return entries unchanged.
        """
        return entries

    # -------- Run orchestration --------
    def run(self, df: pd.DataFrame, run_config: Optional[RunConfig] = None) -> RunResult:
        """
        Execute the strategy given a price DataFrame (UTC index recommended).
        This base implementation focuses on hook wiring + required outputs.
        """
        run_config = run_config or {}
        cash = float(run_config.get("cash", 100_000.0))

        if df is None or len(df) == 0:
            # Construct a single-point UTC series to satisfy acceptance
            now = pd.Timestamp.utcnow().floor("min").tz_localize("UTC")
            idx = pd.DatetimeIndex([now], tz="UTC")
        else:
            idx = _ensure_utc_index(df.index)

        # Defaults
        trades = _empty_trades_df()
        equity = _flat_equity(idx, cash)

        # Determine whether ML hooks are overridden
        uses_ml = (
            self.propose_entries.__func__ is not StrategyBase.propose_entries or
            self.filter_entries.__func__ is not StrategyBase.filter_entries
        )

        proposed_count = 0
        filtered_count = 0
        placed_orders = 0

        if uses_ml:
            entries = self.propose_entries(df)
            entries_df = self._normalize_entries(entries)
            proposed_count = 0 if entries_df is None else int(entries_df.shape[0])

            filtered = self.filter_entries(entries_df, meta_probs=None)
            filtered_df = self._normalize_entries(filtered)
            filtered_count = 0 if filtered_df is None else int(filtered_df.shape[0])

            # This base wiring does not convert entries -> actual trades.
            # Subclasses or Phase 4 runners can add execution. We just log counts.
            placed_orders = 0

            self.logger.info(f"proposed_entries={proposed_count}")
            self.logger.info(f"filtered_entries={filtered_count}")
            self.logger.info(f"placed_orders={placed_orders}")

            # No trades path still valid: return flat equity + empty trades.
            result: RunResult = {"equity": equity, "trades": trades, "stats": {"proposed": proposed_count, "filtered": filtered_count, "placed": placed_orders}}
            return result

        # Else: rule-based path (fallback)
        sig = self.generate_signals(df)
        if sig is not None:
            sig = self._normalize_series(sig)
            proposed_count = int(sig.sum()) if sig is not None else 0
            filtered_count = proposed_count
            placed_orders = 0  # base class does not execute

            self.logger.info(f"proposed_entries={proposed_count}")
            self.logger.info(f"filtered_entries={filtered_count}")
            self.logger.info(f"placed_orders={placed_orders}")

        return {"equity": equity, "trades": trades, "stats": {"proposed": proposed_count, "filtered": filtered_count, "placed": placed_orders}}

    # -------- Helpers --------
    def _normalize_series(self, s: Optional[pd.Series]) -> Optional[pd.Series]:
        if s is None:
            return None
        if not isinstance(s, pd.Series):
            raise TypeError("Expected a pandas Series for signals/entries.")
        s.index = _ensure_utc_index(pd.DatetimeIndex(s.index))
        return s

    def _normalize_entries(self, entries: Optional[Union[pd.Series, pd.DataFrame]]
                           ) -> Optional[pd.DataFrame]:
        if entries is None:
            return None
        if isinstance(entries, pd.Series):
            df = pd.DataFrame({"is_candidate": entries.astype(bool)})
            df.index = _ensure_utc_index(pd.DatetimeIndex(entries.index))
            return df[df["is_candidate"]]
        if isinstance(entries, pd.DataFrame):
            df = entries.copy()
            df.index = _ensure_utc_index(pd.DatetimeIndex(df.index))
            if "is_candidate" not in df.columns:
                df["is_candidate"] = True
            df["is_candidate"] = df["is_candidate"].astype(bool)
            return df[df["is_candidate"]]
        raise TypeError("entries must be a Series or DataFrame")
