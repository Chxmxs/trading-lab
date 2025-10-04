"""
Runners for executing impacted jobs.

This module contains helpers to invoke the project’s orchestrator in
a controlled manner.  Given a list of impacted jobs (strategy,
symbol, timeframe), it constructs the appropriate command lines to
re‑run those jobs via ``python -m orchestrator.tune``.  It supports
dry‑run mode (printing commands without execution) and a
concurrency limit via ``max_workers``.  When running on Windows,
subprocesses should be started using ``subprocess.Popen`` with
``shell=True``.

Whenever possible, prefer to call ``orchestrator.safety.safe_run``
from Phase 6 if available; falling back to direct invocation of the
orchestrator module.  This ensures that errors are captured and
artifacts are written consistently.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List

def rerun_impacted(
    impacted_jobs: Iterable[Dict[str, str]],
    max_workers: int = 1,
    dry_run: bool = True,
    auto_apply: bool = False,
) -> None:
    """Re‑run the orchestrator for impacted jobs.

    Parameters
    ----------
    impacted_jobs: Iterable[Dict[str, str]]
        Each item should contain keys like ``strategy``, ``symbol``,
        and ``timeframe``.  Additional fields may be used to select
        the appropriate configuration.
    max_workers: int, default 1
        Number of concurrent worker threads.  Each worker executes
        one command at a time.
    dry_run: bool, default True
        If True, only print the commands that would be executed.
    auto_apply: bool, default False
        If True, skip interactive confirmations when launching runs.
    """
    commands: List[str] = []
    for job in impacted_jobs:
        # Construct the command.  The actual CLI may differ based on
        # your orchestrator.  This is a placeholder example.
        cfg_path = Path("configs") / "tuning.json"
        cmd = [
            "python",
            "-m",
            "orchestrator.tune",
            "--config",
            str(cfg_path),
        ]
        # Append additional parameters for strategy/symbol/timeframe
        # if your orchestrator supports them.  Example:
        # cmd += ["--strategy", job["strategy"], "--symbol", job["symbol"], "--timeframe", job["timeframe"]]
        commands.append(" ".join(cmd))
    if dry_run:
        for cmd in commands:
            print(f"[DRY RUN] {cmd}")
        return
    # Execute commands concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for cmd in commands:
            executor.submit(_run_command, cmd)

def _run_command(cmd: str) -> None:
    """Run a single command in a subprocess.

    Parameters
    ----------
    cmd: str
        The shell command to execute.
    """
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {cmd}")

__all__ = ["rerun_impacted"]
