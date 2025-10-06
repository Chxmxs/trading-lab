# -*- coding: utf-8 -*-
"""
Phase 10: executor with Data-Health preflight (cached) + SKIPPED(health) MLflow record.

This file runs ONE backtest job based on a "context" dict.

Execution priority:
1) If context["job_callable"] is a function, call it and return its result.
2) Else, if context["strategy_obj"] has .run(df, run_config), call it (Phase 3.5 contract).
   - StrategyBase.run(df, run_config) -> (equity: pd.Series name='equity', trades: pd.DataFrame)
3) Optionally, if context["metrics_callable"] exists, call it after strat.run for Phase 5 artifact writing.

Data-Health integration (new):
- Before any heavy work, run a cached preflight via companion.data_checks.health_runner.preflight_data_health
- Inputs for the cache key: (symbol, timeframe, window{start,end}, dataset-signature of sources)
- If status == "fail": write a minimal MLflow record tagged SKIPPED (health) and return early.
- If status in {"pass","warn"}:
    - If an MLflow run is ALREADY ACTIVE, tag & log immediately (exact lines shown below).
    - Otherwise stash in context so the caller can tag/log right after it opens the real MLflow run:
        context["data_health_status"] = "pass"|"warn"
        context["data_health_summary_path"] = "<cached health_summary.json path>"

No mlfinlab imports here.
"""

from __future__ import annotations
from typing import Any, Dict, List

# === MLflow helpers exposed here so callers can reuse them ====================
from orchestrator.mlflow_utils import (
    start_minimal_run,
    tag_run,
    log_json_artifact,
    mark_skipped_health,
)

# === NEW: Data-health preflight ==============================================
from companion.data_checks.health_runner import preflight_data_health

def _coerce_sources_from_context(ctx: Dict[str, Any]) -> List[str]:
    """Collect the ACTUAL file paths that feed this run.
    Adjust this in one place if you add new sources."""
    out: List[str] = []

    # Common explicit keys:
    for key in ("price_csv", "sopr_csv"):
        p = ctx.get(key)
        if isinstance(p, str) and p:
            out.append(p)

    # Aggregated lists (e.g., built by your context builder):
    agg = ctx.get("data_sources") or ctx.get("metric_files") or ctx.get("sources")
    if isinstance(agg, (list, tuple)):
        for p in agg:
            if isinstance(p, str) and p:
                out.append(p)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

def _extract_window(ctx: Dict[str, Any]) -> Dict[str, str]:
    """Normalize run window to {'start': str(..), 'end': str(..)}."""
    start_dt = ctx.get("start_dt") or ctx.get("run_start") or ctx.get("oos_start") or ctx.get("start")
    end_dt   = ctx.get("end_dt")   or ctx.get("run_end")   or ctx.get("oos_end")   or ctx.get("end")
    return {"start": str(start_dt), "end": str(end_dt)}

def _extract_symbol_tf(ctx: Dict[str, Any]) -> (str, str):
    """Pull symbol/timeframe from context."""
    sym = ctx.get("symbol") or ctx.get("sym") or ctx.get("ticker")
    tf  = ctx.get("timeframe") or ctx.get("tf") or ctx.get("timeframe_str")
    return str(sym), str(tf)

# Optional helper: tag active MLflow run if one is already open
def _maybe_tag_active_mlflow_with_health(context: Dict[str, Any]) -> None:
    try:
        import mlflow  # local import to avoid hard dep if caller handles runs elsewhere
        if mlflow.active_run() is not None:
            status = context.get("data_health_status")
            path = context.get("data_health_summary_path")
            if status and path:
                # >>> EXACT LINES YOU ASKED FOR — RIGHT HERE <<<
                tag_run({"data_health": status})
                log_json_artifact(path, artifact_path="health")
                # mark so callers don't double-tag
                context["_health_tagged"] = True
    except Exception:
        # Non-fatal: tagging is best-effort here; callers can still do it later.
        pass

# ==============================================================================

def run_one_core(context: Dict[str, Any]) -> Any:
    # === DATA HEALTH PREFLIGHT (runs before any heavy work) ====================
    symbol, timeframe = _extract_symbol_tf(context)
    window = _extract_window(context)
    health_sources = _coerce_sources_from_context(context)

    _health = preflight_data_health(
        symbol=symbol,
        timeframe=timeframe,
        window=window,
        sources=health_sources,
        cache_root="artifacts/_cache/data_health",
    )

    if _health["status"] == "fail":
        # Leave a minimal MLflow run so the AI loop can react.
        run_name = f"{symbol}@{timeframe} [SKIPPED:health]"
        start_minimal_run(run_name=run_name, tags={
            "data_health": "fail",
            "skip_reason": "health_check",
            "symbol": symbol,
            "timeframe": timeframe,
        })
        mark_skipped_health()
        log_json_artifact(_health["summary_path"], artifact_path="health")
        # Early out — treated as SKIPPED
        return {
            "skipped": True,
            "reason": "data_health_fail",
            "health_cache_key": _health["cache_key"],
            "symbol": symbol,
            "timeframe": timeframe,
        }

    # For pass|warn, stash info; caller can tag/log once it opens the real MLflow run.
    context["data_health_status"] = _health["status"]            # "pass" or "warn"
    context["data_health_summary_path"] = _health["summary_path"]

    # If (for any reason) an MLflow run is already active at this point, tag NOW:
    _maybe_tag_active_mlflow_with_health(context)

    # === NORMAL EXECUTION PATHS =================================================

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

        # Log entry-filter acceptance metrics if present
        try:
            import mlflow  # lazy import
            if mlflow.active_run() is not None:
                ef_stats = context.get("entry_filter_stats") or {}
                if ef_stats:
                    mlflow.log_metric("filter_kept",        float(ef_stats.get("kept", 0)))
                    mlflow.log_metric("filter_total",       float(ef_stats.get("total", 0)))
                    mlflow.log_metric("filter_accept_rate", float(ef_stats.get("accept_rate", 0.0)))
                    mlflow.log_metric("filter_threshold",   float(ef_stats.get("threshold", 0.5)))
        except Exception:
            pass


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
