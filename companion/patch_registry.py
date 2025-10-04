"""
patch_registry.py
-----------------
Keeps track of available auto-patch functions.
Each patcher must define patch(df) -> df
"""

import importlib
import logging

logger = logging.getLogger(__name__)

# List of patcher module paths (relative imports)
PATCH_MODULES = [
    "companion.patchers.timestamp_patch"
]

def apply_all_patches(df):
    """
    Sequentially apply all patchers in PATCH_MODULES.
    Returns the patched DataFrame.
    """
    patched_df = df
    for modname in PATCH_MODULES:
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "patch"):
                patched_df = mod.patch(patched_df)
                logger.info("Applied patcher: %s", modname)
            else:
                logger.warning("No 'patch' function in %s", modname)
        except Exception as e:
            logger.exception("Failed applying patch %s: %s", modname, e)
    return patched_df
