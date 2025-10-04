"""
Utilities for patching configuration files.

This module provides helper functions to apply changes to JSON or YAML
configuration files in an atomic manner.  All writes should be
performed by writing to a temporary file first and then moving it
into place.  Patches should be expressed as unified diffs so they
can be reviewed and included in patch plans.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

def patch_json_config(config_path: Path, edits: Dict[str, Any]) -> str:
    """Apply simple edits to a JSON configuration file.

    The function reads the existing JSON file (using UTF‑8 with BOM
    support), applies the provided edits at the top level, and writes
    the updated JSON back to disk atomically.  A unified diff of the
    change is returned for inclusion in the patch plan.  If
    ``edits`` is empty, the original file is left unchanged and an
    empty string is returned.

    Parameters
    ----------
    config_path: Path
        Path to the JSON file to edit.
    edits: Dict[str, Any]
        Mapping of keys to new values.  Nested edits are not
        supported in this simple implementation.

    Returns
    -------
    str
        A unified diff representing the change made to the file.
    """
    # Read existing configuration with BOM support
    with config_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    original = json.dumps(data, indent=2, sort_keys=True).splitlines(keepends=True)
    modified = data.copy()
    modified.update(edits)
    new_text = json.dumps(modified, indent=2, sort_keys=True)
    new_lines = new_text.splitlines(keepends=True)
    # Generate unified diff
    import difflib
    diff = difflib.unified_diff(original, new_lines, fromfile=str(config_path), tofile=str(config_path))
    diff_text = "".join(diff)
    if edits:
        # Write changes atomically
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
            tmp.write(new_text)
            tmp_path = Path(tmp.name)
        tmp_path.replace(config_path)
    return diff_text

__all__ = ["patch_json_config"]
