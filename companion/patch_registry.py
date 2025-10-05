# -*- coding: utf-8 -*-
"""
patch_registry.py — centralized registry of safe auto-patch functions.

Each patcher must live in companion/patchers/ and expose a callable:
    def apply_patch(run_id: str, artifacts_dir: Path) -> bool

This module dynamically imports all .py files in companion/patchers/
(except __init__.py) and collects any `apply_patch` functions it finds.
"""

import importlib
import inspect
from pathlib import Path
from typing import Callable, Dict

from companion.logging_config import configure_logging

log = configure_logging(__name__)

PATCHERS_DIR = Path(__file__).parent / "patchers"

def get_patchers() -> Dict[str, Callable]:
    """Discover and return available patchers."""
    patchers = {}
    if not PATCHERS_DIR.exists():
        log.warning("Patchers directory not found: %s", PATCHERS_DIR)
        return patchers

    for file in PATCHERS_DIR.glob("*.py"):
        if file.name.startswith("__"):
            continue
        mod_name = f"companion.patchers.{file.stem}"
        try:
            mod = importlib.import_module(mod_name)
            for name, obj in inspect.getmembers(mod):
                if callable(obj) and name == "apply_patch":
                    patchers[file.stem] = obj
                    log.info("Registered patcher: %s", file.stem)
        except Exception as e:
            log.error("Failed to import patcher %s: %s", mod_name, e)

    if not patchers:
        log.warning("No patchers found in %s", PATCHERS_DIR)
    return patchers

if __name__ == "__main__":
    p = get_patchers()
    print("Discovered patchers:", list(p.keys()))
# --- Legacy compatibility ----------------------------------------------------
def apply_all_patches(run_id: str = None, artifacts_dir=None) -> dict:
    """
    Compatibility wrapper for older modules expecting apply_all_patches().
    It simply discovers and runs all registered patchers sequentially.

    Args:
        run_id: optional MLflow run id (for logging)
        artifacts_dir: optional path (default: artifacts/_quarantine/<run_id>)
    Returns:
        dict: summary { "applied": [...], "errors": [...] }
    """
    from pathlib import Path
    from companion.logging_config import configure_logging
    log = configure_logging(__name__)
    result = {"applied": [], "errors": []}
    try:
        patchers = get_patchers()
        if not patchers:
            log.warning("No patchers registered.")
            return result
        for name, func in patchers.items():
            try:
                ok = func(run_id=run_id or "manual", artifacts_dir=artifacts_dir or Path("artifacts/_quarantine"))
                if ok:
                    result["applied"].append(name)
                    log.info("apply_all_patches: applied %s", name)
                else:
                    result["errors"].append(f"{name}: returned False")
            except Exception as e:
                result["errors"].append(f"{name}: {e}")
                log.error("apply_all_patches: %s failed -> %s", name, e)
    except Exception as e:
        result["errors"].append(str(e))
        log.error("apply_all_patches error: %s", e)
    return result
# ----------------------------------------------------------------------------- 
