# -*- coding: utf-8 -*-
"""
Phase 10 – Queue Consumer (hardened)
Pops jobs from artifacts/_queue/jobs.jsonl and executes them through
orchestrator.execute.run_one_core(), tagging results in MLflow.

Usage:
    python -m companion.ai_loop.consumer --queue artifacts/_queue/jobs.jsonl
    python -m companion.ai_loop.consumer --loop --sleep 5
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# --- Make repo importable regardless of CWD -----------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Core imports -------------------------------------------------------------
import mlflow
from orchestrator.execute import run_one_core, _maybe_tag_active_mlflow_with_health
from orchestrator.mlflow_utils import tag_run, log_json_artifact
from orchestrator.mlflow_bootstrap import init_mlflow

DEFAULT_QUEUE_PATH = REPO_ROOT / "artifacts" / "_queue" / "jobs.jsonl"


# ==============================================================================
# Normalizers & builders
# ==============================================================================

def _normalize_job(job_raw: Any) -> Optional[Dict[str, Any]]:
    """
    Accept tuple/list (job, ...), JSON string/bytes, or dict; return dict or None.
    """
    # Unwrap tuple/list
    if isinstance(job_raw, (list, tuple)) and len(job_raw) > 0:
        job_raw = job_raw[0]

    # Bytes/str -> try JSON
    if isinstance(job_raw, (bytes, bytearray)):
        try:
            job_raw = job_raw.decode("utf-8", errors="replace")
        except Exception:
            job_raw = str(job_raw)

    if isinstance(job_raw, str):
        try:
            return json.loads(job_raw)
        except Exception:
            return None

    # Dict is fine
    if isinstance(job_raw, dict):
        return job_raw

    return None


def _build_strategy_from_job(job: Dict[str, Any]):
    """Construct a strategy object from job spec."""
    if job.get("strategy_stub") == "smoke":
        return type("SmokeStrategy", (), {
            "run": lambda self, df, cfg: (
                __import__("pandas").Series([1, 2, 3], name="equity"),
                __import__("pandas").DataFrame({"ts": [1]})
            )
        })()

    mod = job.get("strategy_module")
    cls = job.get("strategy_class")
    if mod and cls:
        m = importlib.import_module(mod)
        Strategy = getattr(m, cls)
        params = job.get("params") or {}
        try:
            return Strategy(**params)
        except TypeError:
            return Strategy()

    return None


def _context_from_job(job_raw: Any) -> Dict[str, Any]:
    """
    Translate a queue job (any raw form) into the run_one_core context.
    Supports:
      - kind="context" with full "context" dict
      - kind in {"context_min","strategy"} with symbol/tf/window/files + strategy spec
    """
    job = _normalize_job(job_raw)
    if not isinstance(job, dict):
        raise ValueError("Unsupported job format; expected dict/JSON/tuple-of-dict")

    kind = job.get("kind") or ""
    if kind == "context" and isinstance(job.get("context"), dict):
        return dict(job["context"])

    if kind in ("context_min", "strategy"):
        symbol = job.get("symbol") or "UNKNOWN"
        timeframe = job.get("timeframe") or job.get("tf") or "UNKNOWN"
        start = job.get("start") or job.get("start_dt")
        end = job.get("end") or job.get("end_dt")
        price_csv = job.get("price_csv")
        data_sources = job.get("data_sources") or ([price_csv] if price_csv else [])
        strategy_obj = _build_strategy_from_job(job)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_dt": start,
            "end_dt": end,
            "price_csv": price_csv,
            "data_sources": data_sources,
            "strategy_obj": strategy_obj,
            "df": None,
            "run_config": job.get("run_config") or {},
        }

    raise ValueError("Unsupported job schema; expected kind in {'context','context_min','strategy'}")


def _derive_run_name(job_raw: Any) -> str:
    job = _normalize_job(job_raw) or {}
    sym = job.get("symbol") or "UNK"
    tf = job.get("timeframe") or job.get("tf") or "UNK"
    tag = job.get("strategy_class") or job.get("strategy_stub") or "context"
    return f"QCONSUME: {tag} {sym}@{tf}"


# ==============================================================================
# Queue loader
# ==============================================================================

def _load_jobqueue(queue_path: Path):
    """
    Try to load your real JobQueue from companion.ai_loop.queue.
    Fallback to a tiny JSONL impl with .pop_job().
    """
    try:
        qmod = importlib.import_module("companion.ai_loop.queue")
        if hasattr(qmod, "JobQueue"):
            return ("class", qmod.JobQueue(str(queue_path)))
        if hasattr(qmod, "pop_job"):
            return ("func", qmod)
    except Exception:
        pass

    class JSONLQueue:
        def __init__(self, path: Path):
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self.path.write_text("", encoding="utf-8")

        def pop_job(self):
            lines = []
            if self.path.exists():
                lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines:
                return None
            first = lines[0]
            rest = lines[1:]
            self.path.write_text("\n".join(rest) + ("\n" if rest else ""), encoding="utf-8")
            # return raw (string) so normalizer can handle it
            return first

    return ("fallback", JSONLQueue(queue_path))


# ==============================================================================
# Consumer core
# ==============================================================================

def consume_once(queue_path: Path = DEFAULT_QUEUE_PATH) -> int:
    """Pop one job from queue_path and execute it."""
    kind, q = _load_jobqueue(queue_path)

    # Pop raw job (could be tuple, str, bytes, dict)
    if kind == "class":
        job_raw = q.pop_job()
    elif kind == "func":
        job_raw = q.pop_job(queue_path=str(queue_path))
    else:
        job_raw = q.pop_job()

    print(f"[consumer-debug] raw pop_job() type={type(job_raw)}, value={repr(job_raw)[:200]}")


    job = _normalize_job(job_raw)
    if not job:
        print("[consumer] No jobs available or unsupported job format.")
        return 2

    run_name = _derive_run_name(job)

    # Ensure MLflow tracking/experiment for this process
    init_mlflow()

    with mlflow.start_run(run_name=run_name):
        # Helpful tags for lineage (defensive: never assume dict)
        try:
            tag_run({
                "source": "queue_consumer",
                "queue_path": str(queue_path),
                "job_kind": job.get("kind", "") if isinstance(job, dict) else "",
                "symbol": job.get("symbol", "") if isinstance(job, dict) else "",
                "timeframe": job.get("timeframe", "") if isinstance(job, dict) else "",
                "strategy_stub": job.get("strategy_stub", "") if isinstance(job, dict) else "",
                "strategy_module": job.get("strategy_module", "") if isinstance(job, dict) else "",
                "strategy_class": job.get("strategy_class", "") if isinstance(job, dict) else "",
            })
        except Exception:
            pass

        try:
            context = _context_from_job(job)
        except Exception as e:
            tb = traceback.format_exc()
            tmp = REPO_ROOT / ".mlflow_tmp_bad_job.txt"
            tmp.write_text(f"{e}\n\n{tb}\nRAW:\n{repr(job_raw)}", encoding="utf-8")
            log_json_artifact(str(tmp), artifact_path="errors", filename="bad_job.txt")
            tag_run({"run_status": "FAILED", "error_class": "BadJob", "error_msg": str(e)})
            print("[consumer] ERROR: bad job format:", e)
            return 1

        try:
            result = run_one_core(context)
            _maybe_tag_active_mlflow_with_health(context)

            if isinstance(result, dict) and result.get("skipped"):
                tag_run({"run_status": "SKIPPED", "skip_reason": result.get("reason", "")})
                print("[consumer] Run skipped:", result.get("reason"))
                return 0
            else:
                tag_run({"run_status": "OK"})
                print("[consumer] Run finished OK:", result)
                return 0

        except Exception as e:
            tb = traceback.format_exc()
            tmp = REPO_ROOT / ".mlflow_tmp_consumer_exc.txt"
            tmp.write_text(tb, encoding="utf-8")
            log_json_artifact(str(tmp), artifact_path="errors", filename="consumer_exception.txt")
            tag_run({"run_status": "FAILED", "error_class": e.__class__.__name__})
            print("[consumer] ERROR:", e)
            return 1


def main(argv: Optional[list] = None):
    import argparse
    p = argparse.ArgumentParser("Queue consumer")
    p.add_argument("--queue", default=str(DEFAULT_QUEUE_PATH), help="Path to jobs.jsonl")
    p.add_argument("--loop", action="store_true", help="Keep consuming forever (sleeping between attempts)")
    p.add_argument("--sleep", type=int, default=5, help="Seconds to sleep between attempts when looping")
    args = p.parse_args(argv)

    qpath = Path(args.queue)
    qpath.parent.mkdir(parents=True, exist_ok=True)

    if not args.loop:
        ec = consume_once(qpath)
        sys.exit(ec)

    print("[consumer] Loop mode. Queue:", qpath)
    while True:
        ec = consume_once(qpath)
        if ec == 2:
            time.sleep(args.sleep)
        else:
            time.sleep(1)


if __name__ == "__main__":
    main()
