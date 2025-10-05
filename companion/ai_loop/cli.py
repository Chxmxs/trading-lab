# -*- coding: utf-8 -*-
"""
AI Loop CLI (PowerShell 5–friendly)

Commands:
  python -m companion.ai_loop.cli watch           # scan MLflow, write monitor_report.json
  python -m companion.ai_loop.cli status          # show queue snapshot
  python -m companion.ai_loop.cli apply-patches --auto [--experiment NAME|ID] [--lookback H] [--dry-run]
  python -m companion.ai_loop.cli apply-patches --run-id RUN_ID [--dry-run]
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mlflow

try:
    import pandas as pd  # optional, for Timestamp handling
except Exception:
    pd = None

from companion.logging_config import configure_logging
from . import monitor, patch_apply, queue  # local modules
from .queue import JobQueue  # auto-enqueue support

log = configure_logging(__name__)

AI_LOOP_DIR = Path("companion/ai_loop")
LOGS_DIR = AI_LOOP_DIR / "logs"
DECISIONS_PATH = AI_LOOP_DIR / "decisions.jsonl"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _now_utc():
    return datetime.now(timezone.utc).isoformat()


def _append_decision(event_type: str, payload: dict):
    DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = {"ts": _now_utc(), "event": event_type, **payload}
    with DECISIONS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def _to_utc_datetime(val):
    """
    Convert MLflow start_time values to aware UTC datetime.
    Handles: pandas.Timestamp, datetime, int/float epoch (ms or s), ISO 8601 str.
    """
    if val is None:
        return None
    if pd is not None and isinstance(val, pd.Timestamp):
        dt = val.to_pydatetime()
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    if isinstance(val, datetime):
        return val.replace(tzinfo=timezone.utc) if val.tzinfo is None else val.astimezone(timezone.utc)
    if isinstance(val, (int, float)):
        sec = float(val) / 1000.0 if val > 1e12 else float(val)
        return datetime.fromtimestamp(sec, tz=timezone.utc)
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        except Exception:
            try:
                n = float(val)
                return _to_utc_datetime(n)
            except Exception:
                return None
    return None


def find_latest_failed_run(experiment: str | None, lookback_hours: int = 72) -> str | None:
    """
    Return run_id of the most recent non-FINISHED run within lookback window.
    If experiment is None, search across all experiments.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    filter_str = "attributes.status != 'FINISHED'"
    order_by = ["attributes.start_time DESC"]

    exp_ids = []
    if experiment:
        for e in mlflow.search_experiments():
            if e.experiment_id == experiment or e.name == experiment:
                exp_ids = [e.experiment_id]
                break
        if not exp_ids:
            log.warning("Experiment '%s' not found; searching all experiments.", experiment)
    else:
        exp_ids = [e.experiment_id for e in mlflow.search_experiments()]

    newest_run = None
    newest_time = None

    for eid in exp_ids:
        runs = mlflow.search_runs(
            experiment_ids=[eid],
            filter_string=filter_str,
            order_by=order_by,
            max_results=100,
        )
        for _, row in runs.iterrows():
            st = row.get("start_time", None)
            start_dt = _to_utc_datetime(st)
            if start_dt is None or start_dt < since:
                continue
            if newest_time is None or start_dt > newest_time:
                newest_time = start_dt
                newest_run = row["run_id"]

    return newest_run


def cmd_watch(args):
    lookback = int(args.lookback or 6)
    report_path = LOGS_DIR / "monitor_report.json"
    log.info("Running monitor: lookback=%sh", lookback)
    summary = monitor.run_monitor(lookback_hours=lookback, output_path=report_path)
    _append_decision("watch", {"lookback_hours": lookback, "report": str(report_path)})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nWrote: {0}".format(report_path))


def cmd_status(_args):
    q = JobQueue()
    snapshot = q.snapshot()
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    _append_decision("status", {"queued": len(snapshot.get("items", []))})


def cmd_apply_patches(args):
    run_id = args.run_id
    if args.auto and not run_id:
        run_id = find_latest_failed_run(
            experiment=args.experiment,
            lookback_hours=int(args.lookback or 72)
        )

    if not run_id:
        print("No run-id provided and no failed runs found with --auto.", file=sys.stderr)
        sys.exit(2)

    log.info("Applying patches for run_id=%s", run_id)
    result = patch_apply.apply_patches_for_run(run_id=run_id, dry_run=args.dry_run)

    # Auto-enqueue after successful patch (unless dry-run)
    try:
        if not args.dry_run:
            JobQueue().add_item({"run_id": run_id, "priority": "failed_first"})
            log.info("Auto-enqueued run_id=%s after patch", run_id)
    except Exception as e:
        log.error("Auto-enqueue failed: %s", e)

    _append_decision("apply_patches", {
        "run_id": run_id,
        "dry_run": bool(args.dry_run),
        "result": result
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser():
    p = argparse.ArgumentParser(prog="companion.ai_loop.cli", description="AI Loop CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("watch", help="Scan MLflow and write monitor_report.json")
    w.add_argument("--lookback", type=int, default=6, help="Hours to look back (default 6)")
    w.set_defaults(func=cmd_watch)

    s = sub.add_parser("status", help="Show queue status")
    s.set_defaults(func=cmd_status)

    a = sub.add_parser("apply-patches", help="Apply patchers to a run and enqueue for re-run")
    a.add_argument("--run-id", help="Run ID (optional if --auto is used)")
    a.add_argument("--auto", action="store_true", help="Pick the newest failed run automatically")
    a.add_argument("--experiment", help="Restrict search to this experiment name or ID")
    a.add_argument("--lookback", type=int, default=72, help="Hours to look back when using --auto (default 72)")
    a.add_argument("--dry-run", action="store_true", help="Show decisions but do not write changes")
    a.set_defaults(func=cmd_apply_patches)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()