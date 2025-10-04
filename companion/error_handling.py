"""
error_handling.py
Minimal, ASCII-only version: capture_errors + retry_with_backoff with quarantine.
"""

import os
import json
import time
import traceback
import logging
from functools import wraps
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None

logger = logging.getLogger(__name__)

QUARANTINE_DIR = Path("artifacts/_quarantine")
MAX_SAMPLES = 500

def _safe_write(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        logger.exception("write failed %s: %s", path, e)

def _dump_context(ctx: dict, run_name: str) -> Path:
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    folder = QUARANTINE_DIR / f"{ts}_{run_name}"
    folder.mkdir(parents=True, exist_ok=True)

    _safe_write(folder / "context.json", json.dumps(ctx, indent=2))
    _safe_write(folder / "traceback.txt", ctx.get("traceback", ""))
    _safe_write(folder / "exception.txt", ctx.get("exception", ""))

    df = ctx.get("data_sample")
    if pd is not None and getattr(pd, "DataFrame", None) is not None and isinstance(df, pd.DataFrame):
        try:
            df.head(MAX_SAMPLES).to_csv(folder / "df_sample.csv", index=True)
        except Exception:
            pass

    logger.error("quarantined failure saved to: %s", folder)
    return folder

def capture_errors(run_name: str = "unknown_run", context_fn=None):
    """Decorator: capture exceptions, dump context to quarantine, return None."""
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                tb = traceback.format_exc()
                ctx = {
                    "exception": str(e),
                    "traceback": tb,
                    "timestamp": datetime.utcnow().isoformat(),
                    "run_name": run_name,
                }
                if callable(context_fn):
                    try:
                        extra = context_fn(*args, **kwargs) or {}
                        if isinstance(extra, dict):
                            ctx.update(extra)
                    except Exception:
                        logger.exception("context capture failed")
                folder = _dump_context(ctx, run_name)
                logger.exception("run %s failed; quarantined at %s", run_name, folder)
                return None
        return wrapper
    return deco

def retry_with_backoff(max_retries: int = 3, base_delay: int = 2):
    """Decorator: retry transient errors with exponential backoff, then quarantine."""
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    wait = base_delay * (2 ** attempt)
                    logger.warning("attempt %d/%d failed (%s); retrying in %ds",
                                   attempt + 1, max_retries, e, wait)
                    time.sleep(wait)
            tb = traceback.format_exc()
            ctx = {
                "exception": str(last_err) if last_err else "unknown error",
                "traceback": tb,
                "timestamp": datetime.utcnow().isoformat(),
                "run_name": func.__name__,
            }
            _dump_context(ctx, func.__name__)
            return None
        return wrapper
    return deco