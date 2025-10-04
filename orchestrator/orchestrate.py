# -*- coding: utf-8 -*-
"""
Phase 4 Orchestrator (patched for Phase 6):
- Integrates error wrapper around per-job execution path.
- Does NOT import mlfinlab.
- Continues after failures; failed jobs are quarantined with artifacts.
This file assumes Phase 4 already defines run_one_core(context) elsewhere,
or provides an execute_job(context) callable. We wrap whichever is available.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, Optional

# Lazy import to avoid circulars unless used
def _get_error_wrapper():
    try:
        from common.error_handling import wrap_with_error_capture
        return wrap_with_error_capture
    except Exception:
        return None

def _resolve_executor() -> Optional[Callable[[Dict[str, Any]], Any]]:
    """Try common entrypoint names from Phase 4. Return a callable(context)."""
    for mod_name, fn_name in [
        ("orchestrator.core", "run_one_core"),
        ("orchestrator.execute", "run_one_core"),
        ("orchestrator.execute", "execute_job"),
        ("orchestrator.run", "run_one_core"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn
        except Exception:
            continue
    return None

def run_job_with_error_capture(context: Dict[str, Any], *, mlflow_cfg: Optional[Dict[str, Any]] = None, max_retries: int = 2):
    """Public utility for the orchestrator's per-job path."""
    executor = _resolve_executor()
    if executor is None:
        print("[WARN] No executor found to run job. Skipping.")
        return None

    wrapper = _get_error_wrapper()
    if wrapper is None:
        return executor(context)

    return wrapper(lambda: executor(context), context=context, mlflow_cfg=mlflow_cfg, max_retries=max_retries)

if __name__ == "__main__":
    import json
    ctx_path = os.environ.get("TRADEBOT_JOB_CONTEXT_JSON")
    if not ctx_path or not os.path.exists(ctx_path):
        print("[orchestrate] No TRADEBOT_JOB_CONTEXT_JSON provided. Nothing to do.")
        sys.exit(0)
    with open(ctx_path, "r", encoding="utf-8") as f:
        context = json.load(f)
    mlflow_cfg = context.get("mlflow_cfg")
    run_job_with_error_capture(context, mlflow_cfg=mlflow_cfg)