# -*- coding: utf-8 -*-
"""
health.py — basic data health checks used for MLflow tagging and gating runs.
All functions return a tuple: (status: str, details: dict), where status is one of:
  - "pass", "warn", "fail"
"""
from __future__ import annotations

from typing import Tuple, Dict
import pandas as pd

def check_missing_bars(df: pd.DataFrame, timestamp_col: str = "timestamp") -> Tuple[str, Dict]:
    if timestamp_col not in df.columns:
        return "fail", {"reason": f"missing column {timestamp_col}"}
    n = len(df)
    status = "pass" if n > 0 else "fail"
    return status, {"rows": n}

def check_duplicates(df: pd.DataFrame) -> Tuple[str, Dict]:
    dups = int(df.duplicated().sum())
    if dups == 0:
        return "pass", {"duplicates": 0}
    return ("warn" if dups <= 5 else "fail"), {"duplicates": dups}

def check_monotonic_utc(df: pd.DataFrame, timestamp_col: str = "timestamp") -> Tuple[str, Dict]:
    if timestamp_col not in df.columns:
        return "fail", {"reason": f"missing column {timestamp_col}"}
    try:
        ts = pd.to_datetime(df[timestamp_col], utc=True)
    except Exception as e:
        return "fail", {"reason": f"datetime parse failed: {e}"}
    mono = bool(ts.is_monotonic_increasing)
    return ("pass" if mono else "warn"), {"monotonic": mono}

def summarize_health(df: pd.DataFrame) -> Tuple[str, Dict]:
    """
    Aggregate a few checks into a single taggable status.
    fail > warn > pass
    """
    order = {"fail": 2, "warn": 1, "pass": 0}
    statuses = []
    details = {}
    for name, (s, d) in {
        "missing_bars": check_missing_bars(df),
        "duplicates": check_duplicates(df),
        "monotonic": check_monotonic_utc(df),
    }.items():
        statuses.append(s)
        details[name] = d
    worst = sorted(statuses, key=lambda x: order[x], reverse=True)[0]
    return worst, details
