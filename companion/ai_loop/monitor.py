# -*- coding: utf-8 -*-
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import mlflow
    import pandas as pd
except Exception:
    mlflow = None
    pd = None

LOG_DIR = Path("companion/ai_loop/logs")
REPORT_PATH = LOG_DIR / "monitor_report.json"

def _now_iso(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _classify(row: Dict[str, Any]) -> str:
    tb = str(row.get("tags.error_traceback") or row.get("tags.traceback") or "")
    err = str(row.get("tags.error") or "")
    status = str(row.get("status") or "")
    if "schema" in tb.lower() or "schema" in err.lower(): return "schema"
    if "import" in tb.lower(): return "import"
    if any(w in tb.lower() for w in ("file","csv","timestamp")): return "data"
    if status.upper() == "FAILED": return "failed"
    if err: return "error_tagged"
    return "unknown"

def _load_runs(lookback_hours: int) -> List[Dict[str, Any]]:
    if mlflow is None or pd is None: return []
    try:
        df = mlflow.search_runs(max_results=1000)
    except Exception:
        return []
    if df is None or len(df) == 0: return []
    keep = ["run_id","status","experiment_id","start_time","end_time","tags.error","tags.error_traceback","tags.traceback"]
    cols = [c for c in keep if c in df.columns]; df = df[cols]
    now_ms = int(time.time()*1000)
    for c in ("start_time","end_time"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "end_time" in df.columns:
        cutoff = now_ms - (lookback_hours*3600*1000)
        df = df[df["end_time"].fillna(now_ms) >= cutoff]
    mask_failed = (df["status"].fillna("") != "FINISHED") if "status" in df.columns else False
    has_err_tag = (df["tags.error"].astype(str).fillna("") != "") if "tags.error" in df.columns else False
    df = df[mask_failed | has_err_tag]
    items = []
    for _, row in df.iterrows():
        r = row.to_dict(); r["error_type"] = _classify(r); items.append(r)
    return items

def run_watch(lookback_hours: int = 3) -> Dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    items = _load_runs(lookback_hours)
    report = {"generated_at": _now_iso(), "lookback_hours": lookback_hours, "count": len(items), "items": items}
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def get_latest_failed_run_id(lookback_hours: int = 24) -> Optional[str]:
    items = _load_runs(lookback_hours)
    if not items: return None
    def _key(it: Dict[str, Any]) -> int: return int(it.get("end_time") or it.get("start_time") or 0)
    items.sort(key=_key, reverse=True)
    rid = str(items[0].get("run_id") or "")
    return rid or None
