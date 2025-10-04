"""
Utilities for patching Python strategy and tuner code.

Patching source code requires care to avoid breaking semantics.
Whenever possible, prefer configuration changes over code changes.
This module implements minimal, safe modifications using regular
expressions or the ``ast`` module.  High‑risk changes (Tier C) are
guarded and must be confirmed by a human before applying.

Current capabilities include:

* Ensuring that equity outputs from strategies are indexed by a
  pandas.DatetimeIndex with UTC timezone.
* Adding missing columns (e.g. ``exit_time`` and ``pnl_pct``) to
  trades DataFrames.
* Injecting parameter values into run configurations.

Future enhancements might include more sophisticated AST rewriting
using ``lib2to3`` or ``astroid``.  All modifications should be
expressed as unified diffs.
"""

from __future__ import annotations

import ast
import difflib
from pathlib import Path
from typing import Dict, Tuple

def patch_strategy_file(file_path: Path, edits: Dict[str, str]) -> str:
    """Apply simple text substitutions to a strategy source file.

    This helper reads the source file as text, performs literal
    replacements as specified in ``edits``, and writes the result
    atomically.  It returns a unified diff of the changes.  It is
    intended for low‑risk modifications such as adding timezone
    conversions or new columns.  AST‑based rewriting may be added
    later for more complex edits.

    Parameters
    ----------
    file_path: Path
        Path to the Python file to modify.
    edits: Dict[str, str]
        Mapping of literal substrings to their replacements.

    Returns
    -------
    str
        Unified diff of the changes made.  Empty string if no changes.
    """
    original_text = file_path.read_text(encoding="utf-8")
    modified_text = original_text
    for old, new in edits.items():
        modified_text = modified_text.replace(old, new)
    if modified_text == original_text:
        return ""
    diff = difflib.unified_diff(
        original_text.splitlines(keepends=True),
        modified_text.splitlines(keepends=True),
        fromfile=str(file_path),
        tofile=str(file_path),
    )
    diff_text = "".join(diff)
    # Write atomically
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    tmp_path.write_text(modified_text, encoding="utf-8")
    tmp_path.replace(file_path)
    return diff_text

__all__ = ["patch_strategy_file"]
