# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple

# IMPORTANT: import the module, not the function, so monkeypatching works.
health_mod = importlib.import_module("companion.data_checks.health")

def _as_iso_mtime_size(path: str) -> Tuple[str, int]:
    p = Path(path)
    if not p.exists():
        return ("MISSING", -1)
    return (str(int(p.stat().st_mtime)), int(p.stat().st_size))

def _dataset_signature(sources: Iterable[str]) -> Dict[str, Any]:
    sig = {}
    for s in sorted(set([str(x) for x in sources if x])):  # dedupe + order
        p = Path(s)
        mtime = "MISSING"
        size = -1
        if p.exists():
            stat = p.stat()
            mtime = str(int(stat.st_mtime))
            size = int(stat.st_size)
        sig[s] = {"mtime": mtime, "size": size}
    return sig

def _hash_dict(d: Dict[str, Any]) -> str:
    b = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

def _flex_call(fn, **kws):
    """
    Flexibly call summarize_health(...) by matching parameter names.
    Accepts common names: symbol, timeframe, sources/paths, window/start/end.
    Adds a fallback for 'df' by providing None if required.
    """
    sig = inspect.signature(fn)
    args = {}
    params = set(sig.parameters.keys())
    if "symbol" in params and "symbol" in kws: args["symbol"] = kws["symbol"]
    if "timeframe" in params and "timeframe" in kws: args["timeframe"] = kws["timeframe"]
    if "sources" in params and "sources" in kws: args["sources"] = kws["sources"]
    if "paths" in params and "paths" in kws: args["paths"] = kws["paths"]
    if "window" in params and "window" in kws: args["window"] = kws["window"]
    if "start" in params and "start" in kws: args["start"] = kws["start"]
    if "end" in params and "end" in kws: args["end"] = kws["end"]
    # Fallbacks
    if "paths" in params and "paths" not in args and "sources" in kws:
        args["paths"] = kws["sources"]
    # Handle df-required signatures gracefully
    if "df" in params and "df" not in args:
        # If caller passed a df explicitly, include it; otherwise None to satisfy required positional arg
        args["df"] = kws.get("df", None)
    return fn(**args)

def preflight_data_health(
    *,
    symbol: str,
    timeframe: str,
    window: Dict[str, str],
    sources: Iterable[str],
    cache_root: str = "artifacts/_cache/data_health",
) -> Dict[str, Any]:
    """
    Run (or reuse) data health summary, return:
      { "status": "pass|warn|fail", "summary_path": "<json>", "cache_key": "<slug>" }
    """
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    sig = {
        "symbol": symbol,
        "timeframe": timeframe,
        "window": window,
        "dataset": _dataset_signature(sources),
    }
    digest = _hash_dict(sig)
    cache_key = f"{symbol}__{timeframe}__{digest[:16]}"
    out_json = Path(cache_root) / f"{cache_key}.json"

    if out_json.exists():
        with open(out_json, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached["cache_key"] = cache_key
        cached["summary_path"] = str(out_json)
        return cached

    # compute fresh — resolve summarize_health at call time to allow monkeypatches
    summarize_health = getattr(health_mod, "summarize_health")
    summary = _flex_call(
        summarize_health,
        symbol=symbol,
        timeframe=timeframe,
        sources=list(sources),
        window=window,
        start=window.get("start"),
        end=window.get("end"),
    )

    status = (summary.get("status") or summary.get("result") or "").lower().strip()
    if status not in ("pass", "warn", "fail"):
        status = "fail" if summary.get("errors") else "warn" if summary.get("warnings") else "pass"

    payload = {
        "status": status,
        "summary": summary,
        "symbol": symbol,
        "timeframe": timeframe,
        "window": window,
        "cache_key": cache_key,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    payload["summary_path"] = str(out_json)
    return payload