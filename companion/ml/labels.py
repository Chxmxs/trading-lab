# -*- coding: utf-8 -*-
"""
Label generation utilities.

- Triple-barrier (uses mlfinlab when available, else falls back to fixed-horizon logic)
- Fixed-horizon (vectorized, no external deps beyond pandas/numpy)

Outputs Parquet to: data/parquet/<SYMBOL>/<TF>/labels_<scheme>.parquet
Schema (common):
  index (datetime, UTC), columns: ["bin", "ret", "t1", "pt", "sl", "h"]
    bin: {-1,0,1} or {0,1} depending on scheme (we map to {0,1} if side=long-only)
    ret: realized return at t1 relative to entry
    t1 : barrier time or horizon end
    pt/sl: thresholds used
    h: horizon bars (fixed horizon) or vertical barrier bars (triple)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import math

import pandas as pd
import numpy as np

# -------------------- IO --------------------

def _read_price_csv(price_csv: str) -> pd.DataFrame:
    df = pd.read_csv(price_csv)
    # timestamp detection
    ts_col = "timestamp" if "timestamp" in df.columns else "ts" if "ts" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.set_index(ts_col).sort_index()
    # close column detection
    if "close" not in df.columns:
        close_like = [c for c in df.columns if c.lower().startswith("close")]
        if not close_like:
            raise ValueError("CSV missing 'close' column")
        c = close_like[0]
    else:
        c = "close"
    # ensure numeric
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[c])
    df = df[~df.index.duplicated(keep="last")]
    return df[[c]].rename(columns={c: "close"})

def _write_parquet(df: pd.DataFrame, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, engine="pyarrow")
    except Exception:
        # fallback
        df.to_parquet(out_path)

# -------------------- Fixed-horizon (fallback & explicit) --------------------

def fixed_horizon_labels_from_close(
    close: pd.Series,
    *,
    pt: float = 0.02,
    sl: float = 0.02,
    horizon_bars: int = 48,
    side: Optional[str] = "long",  # "long", "short", or None for symmetric
) -> pd.DataFrame:
    """
    For each bar t, look ahead <= horizon_bars, compute forward path returns.
    Label rules:
      - if max_ret >= pt -> +1 (long) / -1 (short)
      - elif min_ret <= -sl -> -1 (long) / +1 (short)
      - else 0
    Return df with ['bin','ret','t1','pt','sl','h'] indexed by t.
    """
    c = close.astype(float).copy()
    idx = c.index
    n = len(c)
    h = int(horizon_bars)

    # Precompute forward returns matrix in a rolling fashion (vectorized-ish)
    # We do a simple loop that's OK for typical sizes; keeps clarity.
    bins = np.zeros(n, dtype=int)
    rets = np.zeros(n, dtype=float)
    t1_ix = np.arange(n)

    for i in range(n):
        j_end = min(n, i + 1 + h)
        if j_end <= i + 1:
            bins[i] = 0
            rets[i] = 0.0
            t1_ix[i] = i
            continue
        window = c.iloc[i+1:j_end].values
        base = float(c.iloc[i])
        if base <= 0 or not np.isfinite(base):
            bins[i] = 0
            rets[i] = 0.0
            t1_ix[i] = i
            continue
        fwd = (window / base) - 1.0
        max_ret = np.nanmax(fwd) if fwd.size else np.nan
        min_ret = np.nanmin(fwd) if fwd.size else np.nan

        # Find first hitting time for either barrier
        hit_j = None
        if np.isfinite(max_ret) and max_ret >= pt:
            # first index where >= pt
            hit_candidates = np.where(fwd >= pt)[0]
            if hit_candidates.size:
                hit_j = i + 1 + int(hit_candidates[0])
                hit_ret = float((c.iloc[hit_j] / base) - 1.0)
                if side == "short":
                    bins[i] = -1  # profit for short is negative barrier; mirror below
                else:
                    bins[i] = 1
                rets[i] = hit_ret
                t1_ix[i] = hit_j
                continue

        if np.isfinite(min_ret) and min_ret <= -sl:
            hit_candidates = np.where(fwd <= -sl)[0]
            if hit_candidates.size:
                hit_j = i + 1 + int(hit_candidates[0])
                hit_ret = float((c.iloc[hit_j] / base) - 1.0)
                if side == "short":
                    bins[i] = 1
                else:
                    bins[i] = -1
                rets[i] = hit_ret
                t1_ix[i] = hit_j
                continue

        # No barrier hit -> end at horizon
        t1_ix[i] = j_end - 1
        end_ret = float((c.iloc[t1_ix[i]] / base) - 1.0)
        bins[i] = 0
        rets[i] = end_ret

    out = pd.DataFrame({
        "bin": bins,
        "ret": rets,
        "t1": idx.values[t1_ix],
        "pt": float(pt),
        "sl": float(sl),
        "h": int(h),
    }, index=idx)
    # For long-only meta-labeling, map {-1,0,1} -> {0,1} by clipping negatives to 0
    if side == "long":
        out["bin"] = (out["bin"] > 0).astype(int)
    elif side == "short":
        out["bin"] = (out["bin"] < 0).astype(int)
    return out

# -------------------- Triple-barrier via mlfinlab (with safe fallback) --------------------

def triple_barrier_labels_from_close(
    close: pd.Series,
    *,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    vol_lookback: int = 50,
    min_ret: Optional[float] = None,
    horizon_bars: int = 48,
    side: Optional[str] = "long",  # long-only meta-label default
) -> pd.DataFrame:
    """
    Attempt mlfinlab triple-barrier. If mlfinlab missing/errors, fallback to fixed-horizon with pt=vol*pt_mult, sl=vol*sl_mult.
    """
    # Try mlfinlab path
    try:
        from mlfinlab.util.volatility import get_daily_vol
        from mlfinlab.filters.cusum_filter import cusum_filter
        from mlfinlab.labeling.labeling import get_events, get_bins
        # Daily vol estimate
        # If index has intraday bars, daily_vol still works (uses daily returns).
        vol = get_daily_vol(close=close, lookback=vol_lookback).reindex(close.index).fillna(method="bfill").fillna(method="ffill")
        if min_ret is None:
            # default minimum return ~ a small fraction of vol
            min_ret = float(vol.median()) * 0.1 if np.isfinite(vol.median()) else 0.001

        # Event timestamps via CUSUM filter (symmetric, threshold ~ min_ret)
        t_events = cusum_filter(close, threshold=min_ret)
        if len(t_events) == 0:
            # fallback: use all bars as events
            t_events = close.index

        # Vertical barrier times
        idx = close.index
        horizon = int(horizon_bars)
        # map each t_event to t + horizon bars (cap at end)
        # Create a dict t1[t] = idx[pos+h-1]
        pos = pd.Series(np.arange(len(idx)), index=idx)
        t1 = {}
        for t in t_events:
            i = int(pos.get(t, -1))
            if i < 0:
                continue
            j = min(len(idx)-1, i + horizon)
            t1[t] = idx[j]
        t1 = pd.Series(t1)

        # pt/sl as multiples of vol at event time
        target = vol
        pt_sl = [float(pt_mult), float(sl_mult)]

        side_arg = None
        if side == "long":
            side_arg = pd.Series(1.0, index=t_events)
        elif side == "short":
            side_arg = pd.Series(-1.0, index=t_events)

        events = get_events(close=close, t_events=pd.Index(t_events), pt_sl=pt_sl,
                            target=target, min_ret=min_ret, num_threads=1,
                            vertical_barrier_times=t1, side=side_arg)

        bins = get_bins(events=events, close=close)
        # bins columns: ['ret','bin','t1'] (mlfinlab)
        out = pd.DataFrame(index=bins.index)
        out["ret"] = bins["ret"].astype(float)
        out["bin"] = bins["bin"].astype(int)
        out["t1"] = bins["t1"].astype("datetime64[ns, UTC]")

        # Normalize to meta-label {0,1} for long-only/short-only
        if side == "long":
            out["bin"] = (out["bin"] > 0).astype(int)
        elif side == "short":
            out["bin"] = (out["bin"] < 0).astype(int)

        out["pt"] = float(pt_mult)
        out["sl"] = float(sl_mult)
        out["h"]  = int(horizon_bars)

        # Reindex to full close index for convenience (optional)
        out = out.reindex(close.index).fillna(method="ffill")
        return out

    except Exception:
        # Fallback: approximate with fixed-horizon using median vol as scale
        vol = close.pct_change().rolling(vol_lookback).std().fillna(0.0)
        scale = float(np.nanmedian(vol)) if np.isfinite(np.nanmedian(vol)) else 0.01
        pt = max(1e-6, float(pt_mult) * scale)
        sl = max(1e-6, float(sl_mult) * scale)
        return fixed_horizon_labels_from_close(close, pt=pt, sl=sl, horizon_bars=horizon_bars, side=side)

# -------------------- High-level entrypoints --------------------

def generate_labels_to_parquet(
    *,
    method: str,  # "triple" or "fixed"
    price_csv: str,
    symbol: str,
    timeframe: str,
    out_root: str = "data/parquet",
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate labels and write to data/parquet/<symbol>/<tf>/labels_<scheme>.parquet.
    Returns output path.
    """
    params = params or {}
    df = _read_price_csv(price_csv)
    close = df["close"].copy()

    if method == "triple":
        out = triple_barrier_labels_from_close(
            close,
            pt_mult=float(params.get("pt_mult", 2.0)),
            sl_mult=float(params.get("sl_mult", 2.0)),
            vol_lookback=int(params.get("vol_lookback", 50)),
            min_ret=params.get("min_ret", None),
            horizon_bars=int(params.get("horizon_bars", 48)),
            side=params.get("side", "long"),
        )
        scheme = f"tb_pt{params.get('pt_mult',2)}_sl{params.get('sl_mult',2)}_h{params.get('horizon_bars',48)}"
    elif method == "fixed":
        out = fixed_horizon_labels_from_close(
            close,
            pt=float(params.get("pt", 0.02)),
            sl=float(params.get("sl", 0.02)),
            horizon_bars=int(params.get("horizon_bars", 48)),
            side=params.get("side", "long"),
        )
        scheme = f"fh_pt{params.get('pt',0.02)}_sl{params.get('sl',0.02)}_h{params.get('horizon_bars',48)}"
    else:
        raise ValueError("method must be 'triple' or 'fixed'")

    out_dir = Path(out_root) / str(symbol) / str(timeframe)
    out_path = out_dir / f"labels_{scheme}.parquet"
    _write_parquet(out, str(out_path))
    return str(out_path)
