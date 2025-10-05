# -*- coding: utf-8 -*-
"""
parallel_run.py — thin wrapper to run a list of CLI tasks in parallel.
Usage:
  python scripts/parallel_run.py --mode mp --cmd "python -m orchestrator.tune --config configs\tuning.json" --repeat 4
"""
from __future__ import annotations
import argparse, subprocess, sys
from multiprocessing import Pool
try:
    import ray
except Exception:
    ray = None

def run_cmd(cmd: str):
    return subprocess.call(cmd, shell=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mp","ray"], default="mp")
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    cmds = [args.cmd for _ in range(args.repeat)]
    if args.mode == "mp":
        with Pool() as p:
            codes = p.map(run_cmd, cmds)
        sys.exit(max(codes) if codes else 0)
    else:
        if ray is None:
            print("ray not installed; falling back to mp")
            return main.__wrapped__()  # type: ignore
        ray.init(ignore_reinit_error=True)
        @ray.remote
        def runc(c): return run_cmd(c)
        futures = [runc.remote(c) for c in cmds]
        codes = ray.get(futures)
        sys.exit(max(codes) if codes else 0)

if __name__ == "__main__":
    main()
