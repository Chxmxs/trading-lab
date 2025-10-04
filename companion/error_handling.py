"""
error_handling.py
-----------------
Centralized error capture, logging, and quarantine system.
Every exception can be wrapped via the `capture_errors` decorator or used manually.
"""

import os
import json
import traceback
import time
import pandas as pd
import logging
from functools import wraps
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
QUARANTINE_DIR = Path("artifacts/_quarantine")
MAX_SAMPLES = 500  # number of rows to save from df slices

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _safe_write(path: Path, content: str):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        logger.exception("Failed to write file %s: %s", path, e)

def _dump_context(context: dict, run_name: str):
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    folder = QUARANTINE_DIR / f"{ts}_{run_name}"
    folder.mkdir(parents=True, exist_ok=True)

    # Save context JSON
    _safe_write(folder / "context.json", json.dumps(context, indent=2))

    # Save traceback
    tb = context.get("traceback", "")
    _safe_write(folder / "traceback.txt", tb)

    # Save exception message
    exc_msg = context.get("exception", "")
    _safe_write(folder / "exception.txt", exc_msg)

    # Optional: save data sample
    df = context.get("data_sample")
    if isinstance(df, pd.DataFrame):
        try:
            df.head(MAX_SAMPLES).to_csv(folder / "df_sample.csv", index=True)
        except Exception:
            pass

    logger.error("Quarantined failure saved to: %s", folder)
    return folder

# --------------------------------------------------------------------
# Decorator: capture errors automatically
# --------------------------------------------------------------------

def capture_errors(run_name="unknown_run", context_fn=None):
    """
    Decorator to wrap any function (e.g., strategy.run or evaluate_strategy_cv)
    and quarantine on error. Optionally supply a context_fn() -> dict
    to capture relevant info (params, dataset slice, etc.)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                tb_text = traceback.format_exc()
                ctx = {
                    "exception": str(e),
                    "traceback": tb_text,
                    "timestamp": datetime.utcnow().isoformat(),
                    "run_name": run_name,
                }
                if callable(context_fn):
                    try:
                        extra = context_fn(*args, **kwargs) or {}
                        ctx.update(extra)
                    except Exception:
                        logger.exception("Context capture failed (ignored).")
                folder = _dump_context(ctx, run_name)
                logger.exception("Run %s failed and was quarantined → %s", run_name, folder)
                return None
        return wrapper
    return decorator

# --------------------------------------------------------------------
# Retry helper for transient errors
# --------------------------------------------------------------------

def retry_with_backoff(max_retries=3, base_delay=2):
    """
    Decorator to retry a transient error several times with exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    wait = base_delay * (2 ** attempt)
                    logger.warning("Attempt %d/%d failed (%s). Retrying in %ds...",
                                   attempt + 1, max_retries, e, wait)
                    time.sleep(wait)
            # If all retries fail
            tb = traceback.format_exc()
            ctx = {
                "exception": str(e),
                "traceback": tb,
                "timestamp": datetime.utcnow().isoformat(),
                "run_name": func.__name__
            }
            _dump_context(ctx, func.__name__)
            return None
        return wrapper
    return decorator
