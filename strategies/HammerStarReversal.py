from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Params:
    lookback:int = 5                 # recent high/low lookback
    wick_ratio:float = 2.0           # long wick >= wick_ratio * body
    max_opp_wick_body:float = 0.3    # "little/no" opposite wick (<= this * body)
    min_body_frac_range:float = 0.05 # body must be at least this fraction of full range
    max_hold:int = 5                 # safety: close after N bars if no flip/stop
    allow_flip:bool = True           # flip on opposite signal


def _load_ohlc(price_csv:str) -> pd.DataFrame:
    df = pd.read_csv(price_csv)
    # standardize columns
    cols = {c.lower():c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    tcol = pick("date","time","timestamp")
    ocol = pick("open","o")
    hcol = pick("high","h")
    lcol = pick("low","l")
    ccol = pick("close","adj_close","c")
    if not all([tcol,ocol,hcol,lcol,ccol]):
        raise ValueError("CSV must have time/open/high/low/close columns.")
    out = df[[tcol,ocol,hcol,lcol,ccol]].copy()
    out.columns = ["time","open","high","low","close"]
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    out = out.dropna(subset=["open","high","low","close"])
    # daily only (enforced by your file choice)
    out = out.set_index("time")
    return out


def _hammer(df:pd.DataFrame, p:Params) -> pd.Series:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng  = (h - l).clip(lower=1e-12)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    body_frac = (body / rng)
    cond_body_min = body_frac >= p.min_body_frac_range
    cond_long_lower = lower >= p.wick_ratio * body
    cond_small_upper = upper <= p.max_opp_wick_body * body
    # "near top": upper wick small already covers this spirit
    # downswing context: new N-day low
    roll_low = l.rolling(p.lookback, min_periods=1).min()
    cond_downswing = l <= roll_low

    return (cond_body_min & cond_long_lower & cond_small_upper & cond_downswing)


def _shooting_star(df:pd.DataFrame, p:Params) -> pd.Series:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng  = (h - l).clip(lower=1e-12)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    body_frac = (body / rng)
    cond_body_min = body_frac >= p.min_body_frac_range
    cond_long_upper = upper >= p.wick_ratio * body
    cond_small_lower = lower <= p.max_opp_wick_body * body
    # "near low": small lower wick already approximates this
    # upswing context: new N-day high
    roll_high = h.rolling(p.lookback, min_periods=1).max()
    cond_upswing = h >= roll_high

    return (cond_body_min & cond_long_upper & cond_small_lower & cond_upswing)


def _apply_fees(px:float, bps:float, side:int, on_entry:bool, slip_bps:float) -> float:
    # side: +1 long, -1 short
    # For long: entry price increases by fees/slippage; exit price decreases.
    # For short: entry price decreases; exit price increases.
    adj = (bps + slip_bps) / 1e4
    if side == +1:
        return px * (1 + adj) if on_entry else px * (1 - adj)
    else:
        return px * (1 - adj) if on_entry else px * (1 + adj)


def _backtest(df:pd.DataFrame, p:Params, fees_bps:float, slippage_bps:float) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Discrete trades. Enter at next bar open after signal.
    Initial stop at pattern extreme (hammer low / star high).
    Trailing stop: each new bar, stop = prior day's low (long) or high (short).
    Exit on stop breach (intrabar) or on opposite signal flip (next open).
    """
    long_sig  = _hammer(df, p)
    short_sig = _shooting_star(df, p)

    idx = df.index
    openp, highp, lowp, closep = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    trades: List[Dict[str,Any]] = []
    equity = []
    eq_val = 1.0

    pos = 0    # +1 long, -1 short, 0 flat
    entry_i = None
    entry_px = None
    stop = None
    entry_extreme = None  # hammer low or star high
    bars_held = 0

    # precompute "enter on next open"
    for i in range(len(df)-1):  # last bar can't trigger next-day entry
        # flip signal computed on bar i, execution on bar i+1
        enter_long  = (pos == 0 and bool(long_sig.iloc[i]))
        enter_short = (pos == 0 and bool(short_sig.iloc[i]))

        # flip case: if in a position and opposite signal occurs, we will close at next open and reverse
        flip_to_long  = (pos == -1 and bool(long_sig.iloc[i]) and p.allow_flip)
        flip_to_short = (pos == +1 and bool(short_sig.iloc[i]) and p.allow_flip)

        # first, manage exits on current bar i (for position from earlier)
        if pos != 0:
            # trailing stop update: stop equals prior day's low/high (use yesterday, i-1)
            if i > (entry_i if entry_i is not None else 0):
                if pos == +1:
                    stop = max(stop, df["low"].iloc[i-1])  # move up to prior day's low
                else:
                    stop = min(stop, df["high"].iloc[i-1]) # move down to prior day's high

            # Check intrabar stop breach on bar i (since we only have daily OHLC):
            exit_hit = False
            exit_px = None
            if pos == +1:
                if lowp[i] <= stop:
                    exit_hit = True
                    exit_px = _apply_fees(stop, fees_bps, +1, on_entry=False, slip_bps=slippage_bps)
            else:
                if highp[i] >= stop:
                    exit_hit = True
                    exit_px = _apply_fees(stop, fees_bps, -1, on_entry=False, slip_bps=slippage_bps)

            bars_held += 1
            # time stop (safety)
            if (not exit_hit) and p.max_hold and bars_held >= p.max_hold:
                exit_hit = True
                # exit at next open if time stop; but we’ll approximate with current close plus fees
                px = closep[i]
                exit_px = _apply_fees(px, fees_bps, pos, on_entry=False, slip_bps=slippage_bps)

            if exit_hit:
                # book trade ending at bar i
                ep = entry_px
                xp = exit_px
                ret = (xp / ep - 1.0) if pos == +1 else (ep / xp - 1.0)
                eq_val *= (1.0 + ret)
                trades.append({
                    "entry_time": idx[entry_i], "exit_time": idx[i],
                    "entry_price": ep, "exit_price": xp, "side": pos,
                    "pnl_pct": ret
                })
                pos = 0
                entry_i = entry_px = stop = entry_extreme = None
                bars_held = 0

        # now, process flips at next open (i+1) or fresh entries if flat
        if flip_to_long or flip_to_short or enter_long or enter_short:
            j = i + 1
            # if still in position (no stop exit happened), close then reverse/open
            if pos != 0:
                # close first at next open
                px = openp[j]
                px = _apply_fees(px, fees_bps, pos, on_entry=False, slip_bps=slippage_bps)
                ret = (px / entry_px - 1.0) if pos == +1 else (entry_px / px - 1.0)
                eq_val *= (1.0 + ret)
                trades.append({
                    "entry_time": idx[entry_i], "exit_time": idx[j],
                    "entry_price": entry_px, "exit_price": px, "side": pos,
                    "pnl_pct": ret
                })
                pos = 0
                entry_i = entry_px = stop = entry_extreme = None
                bars_held = 0

            # open new position
            if flip_to_long or enter_long:
                pos = +1
                entry_i = j
                raw = openp[j]
                entry_px = _apply_fees(raw, fees_bps, +1, on_entry=True, slip_bps=slippage_bps)
                entry_extreme = df["low"].iloc[i]  # hammer's low from signal bar i
                stop = float(entry_extreme)
                bars_held = 0
            elif flip_to_short or enter_short:
                pos = -1
                entry_i = j
                raw = openp[j]
                entry_px = _apply_fees(raw, fees_bps, -1, on_entry=True, slip_bps=slippage_bps)
                entry_extreme = df["high"].iloc[i] # star's high from signal bar i
                stop = float(entry_extreme)
                bars_held = 0

        equity.append((idx[i], eq_val))

    # close any open pos at last bar close
    i = len(df)-1
    if pos != 0:
        px = _apply_fees(closep[i], fees_bps, pos, on_entry=False, slip_bps=slippage_bps)
        ret = (px / entry_px - 1.0) if pos == +1 else (entry_px / px - 1.0)
        eq_val *= (1.0 + ret)
        trades.append({
            "entry_time": idx[entry_i], "exit_time": idx[i],
            "entry_price": entry_px, "exit_price": px, "side": pos,
            "pnl_pct": ret
        })
    equity.append((idx[i], eq_val))

    eq = pd.Series([v for _,v in equity], index=pd.to_datetime([t for t,_ in equity], utc=True))
    tr = pd.DataFrame(trades)
    if not tr.empty:
        for col in ("entry_time","exit_time"):
            tr[col] = pd.to_datetime(tr[col], utc=True)
    return eq, tr


class Strategy:
    name = "HammerStarReversal"

    def run(self, df:Optional[pd.DataFrame], run_config:Dict[str,Any]):
        inputs = (run_config or {}).get("inputs", {})
        price_csv = inputs.get("price_csv")
        if df is None:
            if not price_csv or not Path(price_csv).exists():
                raise FileNotFoundError(f"Missing price_csv: {price_csv}")
            df = _load_ohlc(price_csv)

        # slice run window if provided
        start = pd.to_datetime(run_config.get("start"), utc=True, errors="coerce") if run_config else None
        end   = pd.to_datetime(run_config.get("end"),   utc=True, errors="coerce") if run_config else None
        if isinstance(start, pd.Timestamp):
            df = df.loc[df.index >= start]
        if isinstance(end, pd.Timestamp):
            df = df.loc[df.index <= end]

        p = Params(
            lookback = int(run_config.get("params",{}).get("lookback", 5)),
            wick_ratio = float(run_config.get("params",{}).get("wick_ratio", 2.0)),
            max_opp_wick_body = float(run_config.get("params",{}).get("max_opp_wick_body", 0.3)),
            min_body_frac_range = float(run_config.get("params",{}).get("min_body_frac_range", 0.05)),
            max_hold = int(run_config.get("params",{}).get("max_hold", 5)),
            allow_flip = bool(run_config.get("params",{}).get("allow_flip", True)),
        )
        fees_bps = float((run_config or {}).get("fees_bps", 0))
        slippage_bps = float((run_config or {}).get("slippage_bps", 0))

        equity, trades = _backtest(df, p, fees_bps, slippage_bps)
        return {"equity": equity, "trades": trades}


# ---- Optuna space (so tuner uses this if present) ----
def optuna_space(trial):
    return {
        "lookback": trial.suggest_int("lookback", 3, 7),
        "wick_ratio": trial.suggest_float("wick_ratio", 1.8, 3.0),
        "max_opp_wick_body": trial.suggest_float("max_opp_wick_body", 0.05, 0.5),
        "min_body_frac_range": trial.suggest_float("min_body_frac_range", 0.03, 0.25),
        "max_hold": trial.suggest_int("max_hold", 2, 6),
        "allow_flip": trial.suggest_categorical("allow_flip", [True, False]),
    }

# module-level instance for discovery
STRATEGY = Strategy()
