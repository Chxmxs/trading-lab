from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

def _ensure_events(events: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(events.index, pd.DatetimeIndex):
        raise TypeError("events.index must be DatetimeIndex")
    ev = events.copy()
    ev.index = ev.index.tz_localize("UTC") if ev.index.tz is None else ev.index.tz_convert("UTC")
    if "t1" not in ev.columns:
        raise ValueError("events must have column 't1'")
    ev["t1"] = pd.to_datetime(ev["t1"]).map(lambda x: (pd.Timestamp(x).tz_localize("UTC") if pd.Timestamp(x).tz is None else pd.Timestamp(x).tz_convert("UTC")))
    if (ev.index >= ev["t1"]).any():
        raise ValueError("events with t1 <= start")
    return ev.sort_index()

def _contiguous_folds(n: int, n_splits: int) -> List[np.ndarray]:
    if n_splits < 2 or n_splits > n:
        raise ValueError("n_splits must be in [2, n]")
    base, rem = divmod(n, n_splits)
    idx = np.arange(n)
    out = []
    s = 0
    for k in range(n_splits):
        size = base + (1 if k < rem else 0)
        out.append(idx[s:s+size])
        s += size
    return out

def _overlap(a0: pd.Timestamp, a1: pd.Timestamp, b0: pd.Timestamp, b1: pd.Timestamp) -> bool:
    return (a0 < b1) and (b0 < a1)

def _embargo_window(end_ts: pd.Timestamp, td: Optional[pd.Timedelta]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if not td or td <= pd.Timedelta(0):
        return end_ts, end_ts
    return end_ts, end_ts + td

def _purged_train_idx(ev: pd.DataFrame, te_idx: np.ndarray, embargo_td: Optional[pd.Timedelta]) -> np.ndarray:
    n = len(ev)
    keep = np.ones(n, dtype=bool)
    starts = ev.index.values
    ends = ev["t1"].values
    te_s = ev.index.values[te_idx].min()
    te_e = ev["t1"].values[te_idx].max()
    # purge overlaps
    for i in range(n):
        if _overlap(pd.Timestamp(starts[i]), pd.Timestamp(ends[i]), pd.Timestamp(te_s), pd.Timestamp(te_e)):
            keep[i] = False
    # embargo
    lo, hi = _embargo_window(pd.Timestamp(te_e), embargo_td)
    for i in range(n):
        s = pd.Timestamp(starts[i])
        if lo <= s < hi:
            keep[i] = False
    keep[te_idx] = False
    return np.where(keep)[0]

class PurgedCVRunner:
    def __init__(self, n_splits: int = 5, embargo: Optional[str] = None):
        self.n_splits = int(n_splits)
        self.embargo_td = pd.Timedelta(embargo) if embargo else None

    def split(self, events: pd.DataFrame) -> List[Dict[str, Any]]:
        ev = _ensure_events(events)
        n = len(ev)
        te_folds = _contiguous_folds(n, self.n_splits)
        folds = []
        for te in te_folds:
            tr = _purged_train_idx(ev, te, self.embargo_td)
            folds.append({"train": tr.tolist(), "test": te.tolist()})
        return folds

    def validate(self, events: pd.DataFrame, folds: List[Dict[str, Any]]) -> None:
        ev = _ensure_events(events)
        starts = ev.index
        ends = ev["t1"]
        for i, f in enumerate(folds):
            tr = np.array(f["train"], int)
            te = np.array(f["test"], int)
            if np.intersect1d(tr, te).size > 0:
                raise ValueError(f"Fold {i}: train/test intersect")
            te_s = starts[te].min(); te_e = ends[te].max()
            for j in tr:
                if _overlap(starts[j], ends[j], te_s, te_e):
                    raise ValueError(f"Fold {i}: train event {j} overlaps test interval")
            if self.embargo_td and self.embargo_td > pd.Timedelta(0):
                lo, hi = _embargo_window(te_e, self.embargo_td)
                for j in tr:
                    if lo <= starts[j] < hi:
                        raise ValueError(f"Fold {i}: train event {j} inside embargo")

def compute_uniqueness_weights(events: pd.DataFrame) -> pd.Series:
    ev = _ensure_events(events)
    marks = []
    for i, (s, e) in enumerate(zip(ev.index, ev["t1"])):
        marks.append((s, +1, i))
        marks.append((e, -1, i))
    marks.sort(key=lambda x: (x[0], -x[1]))
    active = set()
    acc = np.zeros(len(ev), float)
    dur = np.array([(e - s).total_seconds() for s, e in zip(ev.index, ev["t1"])], float)
    dur[dur == 0] = 1.0
    for (t_i, sign, idx), (t_j, _, _) in zip(marks[:-1], marks[1:]):
        if sign == +1: active.add(idx)
        else: active.discard(idx)
        dt = (t_j - t_i).total_seconds()
        if dt <= 0 or not active: continue
        share = 1.0 / len(active)
        for k in active:
            acc[k] += dt * share
    w = acc / dur
    return pd.Series(np.clip(w, 0.0, 1.0), index=ev.index, name="sample_weight")
