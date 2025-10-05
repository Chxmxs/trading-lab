# -*- coding: utf-8 -*-
"""
patch_apply.py – safely apply registered patchers to a failed run.

This integrates with:
- companion/patch_registry.py (for available patch functions)
- artifacts/_quarantine/ (for failed run bundles)
- MLflow run metadata for context

Supports dry_run=True to preview patch actions without changing files.
"""

import json
from pathlib import Path
import traceback
from datetime import datetime, timezone

import mlflow

from companion.logging_config import configure_logging
from companion.patch_registry import get_patchers  # must exist in your repo

log = configure_logging(__name__)

AI_LOOP_DIR = Path("companion/ai_loop")
DECISIONS_PATH = AI_LOOP_DIR / "decisions.jsonl"
QUARANTINE_DIR = Path("artifacts/_quarantine")
QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

def _append_decision(event_type, payload):
    """Append audit entry to ai_loop/decisions.jsonl."""
    line = {"ts": datetime.now(timezone.utc).isoformat(), "event": event_type, **payload}
    DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DECISIONS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

def apply_patches_for_run(run_id: str, dry_run: bool = False):
    """
    Load failure info for run_id, classify, and apply matching patchers.

    Args:
        run_id: MLflow run identifier.
        dry_run: If True, only report what would be done.
    Returns:
        dict with summary: {run_id, applied, skipped, errors}
    """
    result = {"run_id": run_id, "applied": [], "skipped": [], "errors": []}
    try:
        log.info("Loading MLflow run %s", run_id)
        run = mlflow.get_run(run_id)
        tags = run.data.tags or {}
        status = run.info.status
        artifacts_dir = QUARANTINE_DIR / run_id

        log.info("Run status=%s, tags=%d", status, len(tags))

        # Determine patch classes by tag or trace hints
        patchers = get_patchers()  # returns dict{name: func}
        candidates = []
        for name in patchers.keys():
            if name.lower() in json.dumps(tags).lower():
                candidates.append(name)
        if not candidates:
            # fallback to generic if no hint found
            candidates = list(patchers.keys())

        for name in candidates:
            func = patchers[name]
            try:
                if dry_run:
                    log.info("[dry-run] Would apply patch: %s", name)
                    result["skipped"].append(name)
                    continue
                log.info("Applying patch: %s", name)
                ok = func(run_id=run_id, artifacts_dir=artifacts_dir)
                if ok:
                    result["applied"].append(name)
                else:
                    result["errors"].append(f"{name}: returned False")
            except Exception as e:
                tb = traceback.format_exc(limit=4)
                log.error("Patch %s failed: %s", name, e)
                result["errors"].append(f"{name}: {e}")
                _append_decision("patch_error", {"run_id": run_id, "patch": name, "error": str(e), "trace": tb})

        _append_decision("patch_apply", result)
    except Exception as e:
        tb = traceback.format_exc(limit=4)
        log.error("apply_patches_for_run failed: %s", e)
        result["errors"].append(str(e))
        _append_decision("patch_apply_error", {"run_id": run_id, "error": str(e), "trace": tb})

    return result
