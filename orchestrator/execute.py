# -*- coding: utf-8 -*-
"""
Phase 4 executor (added for Phase 6 integration).
Provides un_one_core(context) so the Phase-6 wrapper can call ONE job safely.

Priority:
1) If context["job_callable"] is a function, call it and return its result.
2) Else, if context["strategy_obj"] has .run(df, run_config) (Phase 3.5 contract), call it.
   - StrategyBase.run(df, run_config) -> (equity: pd.Series name='equity', trades: pd.DataFrame exact schema)
3) Optionally, if context["metrics_callable"] exists, call it after strat.run for Phase 5 artifact writing.

No mlfinlab imports here.
"""

from __future__ import annotations
from typing import Any, Dict

def run_one_core(context: Dict[str, Any]) -> Any:
    # 1) User-provided callable does everything end-to-end
    job = context.get("job_callable")
    if callable(job):
        return job(context)

    # 2) Default: call StrategyBase.run(df, run_config)
    strat = context.get("strategy_obj")
    if strat is not None and hasattr(strat, "run"):
        df = context.get("df")
        run_config = context.get("run_config") or {}
        equity, trades = strat.run(df, run_config)

        # Stash for downstream steps if needed
        context["equity"] = equity
        context["trades"] = trades

        metrics_cb = context.get("metrics_callable")
        if callable(metrics_cb):
            return metrics_cb(context)

        try:
            tcount = len(trades) if trades is not None else 0
        except Exception:
            tcount = None
        return {"status": "ok", "trades_count": tcount}

    # 3) Nothing to execute -> tell caller what is missing
    raise RuntimeError(
        "run_one_core(context) needs either context['job_callable'] (callable) "
        "OR context['strategy_obj'] with .run(df, run_config)."
    )