# strategies/SOPRRegimeBand.py
# SOPR Regime Band Strategy — Phase 3.5 compatible (no mlfinlab)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import os
import numpy as np
import pandas as pd


# ------------------ Params ------------------
@dataclass
class Params:
    band_low: float = 0.98
    band_high: float = 1.02
    band_pad: float = 0.002   # ± pad around band for entry detection
    ma_short: int = 7
    ma_regime: int = 90
    tp_pct: float = 0.06
    sl_pct: float = 0.05
    sopr_tp_bull: float = 1.05
    sopr_tp_bear: float = 0.95
    regime_eps: float = 0.0
    max_hold_days: int = 120


# ------------------ TZ helpers ------------------
def _to_utc_index(idx) -> pd.DatetimeIndex:
    dt = pd.to_datetime(idx, utc=False)
    if not isinstance(dt, pd.DatetimeIndex):
        dt = pd.DatetimeIndex(dt)
    if dt.tz is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    return dt


def _normalize_daily_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dt = idx.normalize()
    if dt.tz is None:
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")
    return dt


# ------------------ IO helpers ------------------
def _load_csv_price(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in ("time", "timestamp", "date", "dt") if c in df.columns), df.columns[0])
    dt = _to_utc_index(df[tcol])
    df = df.copy()
    df.index = dt
    return df


def _load_price_df(run_config: Dict[str, Any]) -> pd.DataFrame:
    inputs = run_config.get("inputs", {}) or {}
    price_csv = inputs.get("price_csv")
    symbol = run_config.get("symbol")
    timeframe = run_config.get("timeframe")
    tried = []

    if price_csv:
        tried.append(price_csv)
        if os.path.exists(price_csv):
            return _load_csv_price(price_csv)

    if symbol and timeframe:
        guess = os.path.join("data", "cleaned", f"ohlcv_{symbol}@{timeframe}.csv")
        tried.append(guess)
        if os.path.exists(guess):
            return _load_csv_price(guess)

    raise ValueError(
        "Price data not found. Provide run_config.inputs.price_csv or ensure "
        "data/cleaned/ohlcv_<symbol>@<timeframe>.csv exists. Tried: " + "; ".join(tried)
    )


def _load_sopr_from_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if str(c).lower() in ("date", "time", "timestamp")), df.columns[0])
    vcol = next((c for c in df.columns if c != tcol and pd.api.types.is_numeric_dtype(df[c])), None)
    if vcol is None:
        raise ValueError("SOPR CSV has no numeric value column.")
    s = pd.Series(df[vcol].values, index=_to_utc_index(df[tcol]), name="sopr").sort_index()
    s = s.groupby(_normalize_daily_utc(s.index)).last()
    s.index.name = None
    return s


def _get_daily_price(df: pd.DataFrame) -> pd.Series:
    close_candidates = [c for c in df.columns if str(c).lower() == "close"]
    if not close_candidates:
        raise ValueError("Input price df must contain a 'close' column.")
    close_name = close_candidates[0]

    if isinstance(df.index, pd.DatetimeIndex):
        dt = _to_utc_index(df.index)
    else:
        tcol = next((c for c in ("time", "timestamp", "date", "dt") if c in df.columns), None)
        if tcol is None:
            raise ValueError("Provide DatetimeIndex or a time/timestamp column on price df.")
        dt = _to_utc_index(df[tcol])

    price = pd.Series(df[close_name].values, index=dt, name="close").sort_index()
    daily = price.groupby(_normalize_daily_utc(price.index)).last()
    daily.index.name = None
    return daily


def _prepare_sopr(price_index: pd.DatetimeIndex, df: Optional[pd.DataFrame], run_config: Dict[str, Any]) -> pd.Series:
    # Prefer SOPR/aSOPR in provided df if present
    if df is not None and hasattr(df, "columns"):
        for c in df.columns:
            lc = str(c).lower()
            if lc in ("sopr", "asopr", "a_sopr"):
                if isinstance(df.index, pd.DatetimeIndex):
                    dt = _to_utc_index(df.index)
                else:
                    tcol = next((k for k in ("time", "timestamp", "date", "dt") if k in df.columns), None)
                    if tcol is None:
                        break
                    dt = _to_utc_index(df[tcol])
                s = pd.Series(df[c].values, index=dt, name="sopr").sort_index()
                s = s.groupby(_normalize_daily_utc(s.index)).last()
                return s.reindex(price_index).ffill(limit=3)

    inputs = run_config.get("inputs", {}) or {}
    sopr_csv = inputs.get("sopr_csv")
    if not sopr_csv:
        raise ValueError("SOPR data missing. Provide 'sopr' column on df or run_config['inputs']['sopr_csv'].")
    s = _load_sopr_from_csv(sopr_csv)
    return s.reindex(price_index).ffill(limit=3)


def _fees_slip_bps(run_config: Dict[str, Any]) -> float:
    return float(run_config.get("fees_bps", 0.0)) + float(run_config.get("slippage_bps", 0.0))


# ------------------ Strategy ------------------
class Strategy:
    """SOPR Regime Band Strategy (long in bull dips; short in bear rallies)."""

    def __init__(self, params: Optional[Params] = None):
        self.params = params or Params()

    def _build_signals(self, price_d: pd.Series, sopr_d: pd.Series) -> pd.DataFrame:
        p = self.params
        df = pd.DataFrame({"close": price_d, "sopr": sopr_d}).dropna()

        df["sopr_ma_short"] = df["sopr"].rolling(p.ma_short, min_periods=p.ma_short).mean()
        df["sopr_ma_regime"] = df["sopr"].rolling(p.ma_regime, min_periods=p.ma_regime).mean()

        df["regime_bull"] = (df["sopr_ma_regime"] > (1.0 + p.regime_eps)).astype(int)
        df["regime_bear"] = (df["sopr_ma_regime"] < (1.0 - p.regime_eps)).astype(int)

        df["sopr_diff"] = df["sopr_ma_short"].diff()

        # in-band with ±pad ("touch & turn")
        df["in_band"] = (
            (df["sopr_ma_short"] >= (p.band_low - p.band_pad)) &
            (df["sopr_ma_short"] <= (p.band_high + p.band_pad))
        ).astype(int)

        df["long_signal"] = (
            (df["regime_bull"] == 1) &
            (df["in_band"] == 1) &
            (df["sopr_diff"] > 0)
        )

        df["short_signal"] = (
            (df["regime_bear"] == 1) &
            (df["in_band"] == 1) &
            (df["sopr_diff"] < 0)
        )

        return df

    def run(self, df: Optional[pd.DataFrame], run_config: Dict[str, Any]) -> Tuple[pd.Series, pd.DataFrame]:
        p = self.params

        # Self-load price if orchestrator passes None
        if df is None:
            df = _load_price_df(run_config)

        price_d = _get_daily_price(df)
        sopr_d = _prepare_sopr(price_d.index, df, run_config)
        sig = self._build_signals(price_d, sopr_d).dropna()

        feeslip_bps = _fees_slip_bps(run_config)
        roundtrip_cost = 2.0 * feeslip_bps / 1e4
        fees_open = roundtrip_cost / 2.0
        fees_close = roundtrip_cost / 2.0

        in_pos = False
        side = None
        entry_px = None
        entry_dt = None
        rows: List[Dict[str, Any]] = []
        equity = pd.Series(index=sig.index, dtype=float)
        equity.iloc[0] = 1.0
        last_equity = 1.0
        held_days = 0

        for i in range(1, len(sig)):
            dt_prev = sig.index[i - 1]
            dt = sig.index[i]
            px_prev = sig["close"].iloc[i - 1]
            px = sig["close"].iloc[i]
            sopr_short = sig["sopr_ma_short"].iloc[i]
            sopr_reg = sig["sopr_ma_regime"].iloc[i]

            if not in_pos:
                if bool(sig["long_signal"].iloc[i]):
                    in_pos, side, entry_px, entry_dt, held_days = True, "long", px, dt, 0
                    last_equity *= (1.0 - fees_open)
                elif bool(sig["short_signal"].iloc[i]):
                    in_pos, side, entry_px, entry_dt, held_days = True, "short", px, dt, 0
                    last_equity *= (1.0 - fees_open)
            else:
                ret = (px / px_prev - 1.0) if side == "long" else (px_prev / px - 1.0)
                last_equity *= (1.0 + ret)
                held_days += 1

                price_change_from_entry = (px / entry_px - 1.0) if side == "long" else (entry_px / px - 1.0)
                exit_reason = None

                if price_change_from_entry >= p.tp_pct:
                    exit_reason = "tp_price"
                elif price_change_from_entry <= -p.sl_pct:
                    exit_reason = "sl_price"

                if exit_reason is None:
                    if side == "long" and sopr_short >= p.sopr_tp_bull:
                        exit_reason = "tp_sopr"
                    elif side == "short" and sopr_short <= p.sopr_tp_bear:
                        exit_reason = "tp_sopr"

                if exit_reason is None:
                    bull_now = sopr_reg > (1.0 + p.regime_eps)
                    bear_now = sopr_reg < (1.0 - p.regime_eps)
                    if side == "long" and bear_now:
                        exit_reason = "regime_flip"
                    elif side == "short" and bull_now:
                        exit_reason = "regime_flip"

                if exit_reason is None and held_days >= p.max_hold_days:
                    exit_reason = "max_hold"

                if exit_reason is not None:
                    last_equity *= (1.0 - fees_close)
                    pnl_pct = (px / entry_px - 1.0 - roundtrip_cost) if side == "long" else (entry_px / px - 1.0 - roundtrip_cost)
                    rows.append({
                        "entry_time": entry_dt,
                        "exit_time": dt,
                        "side": side,
                        "entry_price": float(entry_px),
                        "exit_price": float(px),
                        "pnl_pct": float(pnl_pct),
                        "reason": exit_reason,
                        "bars_held": int(held_days),
                    })
                    in_pos, side, entry_px, entry_dt, held_days = False, None, None, None, 0

            equity.iloc[i] = last_equity

        equity.name = "equity"
        trades = pd.DataFrame(rows)
        for c in ("entry_time", "exit_time"):
            if c in trades.columns:
                trades[c] = pd.to_datetime(trades[c], utc=True)
        return equity, trades


# ------------------ Optuna space ------------------
def optuna_space(trial):
    return {
        "band_low": trial.suggest_float("band_low", 0.965, 1.00),
        "band_high": trial.suggest_float("band_high", 1.00, 1.035),
        "band_pad": trial.suggest_float("band_pad", 0.000, 0.005),
        "ma_short": trial.suggest_int("ma_short", 5, 10),
        "ma_regime": trial.suggest_int("ma_regime", 60, 120, step=5),
        "tp_pct": trial.suggest_float("tp_pct", 0.05, 0.08),
        "sl_pct": trial.suggest_float("sl_pct", 0.04, 0.07),
        "sopr_tp_bull": trial.suggest_float("sopr_tp_bull", 1.03, 1.08),
        "sopr_tp_bear": trial.suggest_float("sopr_tp_bear", 0.92, 0.98),
        "regime_eps": trial.suggest_float("regime_eps", 0.0, 0.01),
        "max_hold_days": trial.suggest_int("max_hold_days", 30, 180, step=10),
    }


# Optional module-level instance (helps discovery)
STRATEGY = Strategy()