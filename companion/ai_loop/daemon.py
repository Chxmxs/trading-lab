# -*- coding: utf-8 -*-
"""
Phase 10 – Intelligence Daemon
Single loop that: monitor -> enqueue patches -> consume queue -> refresh leaderboard -> retention
"""

from __future__ import annotations
import sys, time, importlib, traceback, subprocess
from pathlib import Path
from typing import Optional

# repo bootstrap
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# MLflow bootstrap
from orchestrator.mlflow_bootstrap import init_mlflow
import mlflow

# Consumer
from companion.ai_loop.consumer import consume_once

LOG_DIR = REPO_ROOT / "artifacts" / "_ci"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "daemon.log"

def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    print(line, end="")

def _try_call(module_name: str, fn_candidates: list[str], *args, **kwargs):
    """
    Import module and try a list of function names until one works. Returns (found, result).
    """
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        _log(f"[warn] cannot import {module_name}: {e}")
        return (False, None)

    for fn in fn_candidates:
        if hasattr(mod, fn):
            try:
                res = getattr(mod, fn)(*args, **kwargs)
                return (True, res)
            except Exception as e:
                _log(f"[error] {module_name}.{fn} raised: {e}\n{traceback.format_exc()}")
                return (True, None)
    _log(f"[warn] none of {fn_candidates} found in {module_name}")
    return (False, None)

def scan_and_patch_once() -> int:
    """
    Ask monitor to scan MLflow and enqueue patched jobs.
    Returns number of jobs enqueued (best-effort).
    """
    found, res = _try_call("companion.ai_loop.monitor",
                           ["scan_mlflow_once", "scan_once", "monitor_once"])
    if not found:
        _log("[info] monitor module not available; skipping scan")
        return 0
    # Expect res like {"queued": int, ...} or an int
    try:
        if isinstance(res, dict) and "queued" in res:
            return int(res["queued"])
        if isinstance(res, int):
            return res
    except Exception:
        pass
    return 0

def refresh_leaderboard_once() -> None:
    """
    Refresh master_list.json + leaderboard.md.
    """
    found, _ = _try_call("companion.explorer.refresh_master",
                         ["refresh", "refresh_once", "main"])
    if not found:
        # Fallback: try running as a script if present
        script = REPO_ROOT / "scripts" / "refresh_master.py"
        if script.exists():
            try:
                subprocess.run([sys.executable, str(script)], check=False)
            except Exception as e:
                _log(f"[warn] refresh_master.py subprocess failed: {e}")

def retention_once() -> None:
    """
    Run retention/cache cleanup if module present.
    """
    _try_call("companion.ai_loop.retention",
              ["run_once", "retention_once", "apply_retention"])

def cycle_once(consume_batch: int = 3) -> None:
    # Ensure MLflow tracking/experiment are set
    uri = init_mlflow()
    _log(f"[cycle] using MLflow tracking: {uri}")

    # 1) monitor -> enqueue patches
    try:
        q = scan_and_patch_once()
        _log(f"[cycle] monitor enqueued: {q}")
    except Exception as e:
        _log(f"[error] monitor failed: {e}")

    # 2) consume up to N jobs
    processed = 0
    for _ in range(max(1, int(consume_batch))):
        ec = consume_once()
        if ec == 2:  # empty queue
            break
        processed += 1
    _log(f"[cycle] consumed: {processed}")

    # 3) refresh leaderboard/master
    try:
        refresh_leaderboard_once()
        _log("[cycle] leaderboard refreshed")
    except Exception as e:
        _log(f"[warn] leaderboard refresh failed: {e}")

    # 4) retention
    try:
        retention_once()
        _log("[cycle] retention done")
    except Exception as e:
        _log(f"[warn] retention failed: {e}")

def main(argv: Optional[list] = None):
    import argparse
    p = argparse.ArgumentParser("Companion Intelligence Daemon")
    p.add_argument("--loop", action="store_true", help="Run forever")
    p.add_argument("--sleep", type=int, default=15, help="Seconds to sleep between cycles")
    p.add_argument("--consume-batch", type=int, default=3, help="Jobs to pull per cycle")
    args = p.parse_args(argv)

    if not args.loop:
        cycle_once(consume_batch=args.consume_batch)
        return

    _log("[daemon] starting loop")
    while True:
        try:
            cycle_once(consume_batch=args.consume_batch)
        except Exception as e:
            _log(f"[fatal] cycle crashed: {e}\n{traceback.format_exc()}")
        time.sleep(max(1, int(args.sleep)))

if __name__ == "__main__":
    main()