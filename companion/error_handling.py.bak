$P = "C:\Users\Gladiator\trading-lab\companion\error_handling.py"
# Make a safety backup
if (Test-Path $P) { Copy-Item $P "$P.bak" -Force }

@"
\"\"\"
error_handling.py
Centralized error capture, logging, and quarantine system.
\"\"\"

import json
import logging
import traceback
import time
from functools import wraps
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)

# Locations / limits
QUARANTINE_DIR = Path("artifacts/_quarantine")
MAX_SAMPLES = 500  # rows in df sample

def _safe_write(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        logger.exception("Failed to write %s: %s", path, e)

def _dump_context(ctx: Dict[str, Any], run_name: str) -> Path:
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    folder = QUARANTINE_DIR / f"{ts}_{run_name}"
    folder.mkdir(parents=True, exist_ok=True)

    _safe_write(folder / "context.json", json.dumps(ctx, indent=2))
    _safe_write(folder / "traceback.txt", ctx.get("traceback", ""))
    _safe_write(folder / "exception.txt", ctx.get("exception", ""))

    df = ctx.get("data_sample")
    if isinstance(df, pd.DataFrame):
        try:
            df.head(MAX_SAMPLES).to_csv(folder / "df_sample.csv", index=True)
        except Exception:
            pass

    logger.error("Quarantined failure saved to: %s", folder)
    return folder

def capture_errors(run_name: str = "unknown_run", context_fn: Optional[Callable[..., Dict[str, Any]]] = None):
    \"\"\"Decorator: capture exceptions, dump context, and continue (returns None on failure).\"\"\"
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
                        ctx.update(extra)
                    except Exception:
                        logger.exception("Context capture failed (ignored).")
                folder = _dump_context(ctx, run_name)
                logger.exception("Run %s failed and was quarantined \u2192 %s", run_name, folder)
                return None
        return wrapper
    return deco

def retry_with_backoff(max_retries: int = 3, base_delay: int = 2):
    \"\"\"Decorator: retry transient errors with exponential backoff, then quarantine.\"\"\"
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
                    logger.warning("Attempt %d/%d failed (%s). Retrying in %ds...", attempt+1, max_retries, e, wait)
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
"@ | Out-File -FilePath $P -Encoding utf8 -Force
