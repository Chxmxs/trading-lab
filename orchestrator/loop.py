# -*- coding: utf-8 -*-
"""
Minimal orchestrator loop that:
- Builds a small list of jobs.
- For each job: builds context and calls run_job_with_error_capture().
- Uses a simple job_callable so you can confirm plumbing end-to-end.
Later, replace job_callable with your real strategy_obj + df + run_config.
"""

from __future__ import annotations
import os, sys, json, hashlib, datetime as dt
from typing import Dict, Any, List

# Make sure repo root is on path when run as a script
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from orchestrator.orchestrate import run_job_with_error_capture

def _utc_compact() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def _paramhash(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

def main():
    # --- 1) define tiny list of jobs (dummy for now) ---
    jobs: List[Dict[str, Any]] = [
        {
            "strategy": "SmokeStrat",
            "symbol": "BTCUSD",
            "timeframe": "15m",
            "params": {"lookback": 20, "thresh": 1.5},
            "job_callable": lambda ctx: {"status": "ok", "job": ctx["key"]},
        },
        {
            "strategy": "SmokeStrat",
            "symbol": "ETHUSD",
            "timeframe": "1h",
            "params": {"lookback": 30, "thresh": 2.0},
            "job_callable": lambda ctx: {"status": "ok", "job": ctx["key"]},
        },
    ]

    mlflow_cfg = {"enabled": False, "tracking_uri": None, "run_id": None}

    successes, failures = 0, 0

    for job in jobs:
        strategy, symbol, timeframe, params = job["strategy"], job["symbol"], job["timeframe"], job["params"]
        key = f"{strategy}_{symbol}_{timeframe}"
        ts = _utc_compact()
        ph = _paramhash(params)

        context: Dict[str, Any] = {
            "key": key,
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "params": params,
            "run_config": {"start": "2024-01-01", "end": "2024-01-07"},
            "paramhash": ph,
            "timestamp": ts,
            "artifacts_root": "artifacts",
            "job_callable": job.get("job_callable"),
        }

        print(f"[RUN] {key} params={params}")
        result = run_job_with_error_capture(context, mlflow_cfg=mlflow_cfg, max_retries=2)

        if result is None:
            print(f"[FAIL] {key}")
            failures += 1
        else:
            print(f"[OK]   {key} -> {result}")
            successes += 1

    print(f"\n=== DONE: {successes} ok / {failures} fail ===")

if __name__ == "__main__":
    main()