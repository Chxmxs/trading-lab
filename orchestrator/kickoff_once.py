# -*- coding: utf-8 -*-
from orchestrator.orchestrate import run_job_with_error_capture

def main():
    context = {
        "key": "SmokeTest_KEY",
        "strategy": "SmokeTest",
        "symbol": "BTCUSD",
        "timeframe": "15m",
        "params": {"demo": 1},
        "run_config": {"start": "2024-01-01", "end": "2024-01-02"},
        "paramhash": "demo123",
        "timestamp": "2025-10-02T21:00:00Z",
        "artifacts_root": "artifacts",
        # <-- This makes the test PASS without needing a real strategy/df:
        "job_callable": lambda ctx: {"status": "ok", "note": "hello via job_callable"},
    }

    mlflow_cfg = {"enabled": False, "tracking_uri": None, "run_id": None}
    result = run_job_with_error_capture(context, mlflow_cfg=mlflow_cfg, max_retries=2)
    print("Result:", result)

if __name__ == "__main__":
    main()