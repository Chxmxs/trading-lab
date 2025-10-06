# -*- coding: utf-8 -*-
"""
companion.ai_loop.patterns
Registry of common error signatures and lightweight classifiers.
"""

import re

ERROR_PATTERNS = {
    "data_error": [
        re.compile(r"FileNotFoundError", re.I),
        re.compile(r"missing ohlcv", re.I),
        re.compile(r"schema", re.I),
    ],
    "logic_error": [
        re.compile(r"AssertionError", re.I),
        re.compile(r"TypeError", re.I),
        re.compile(r"ValueError", re.I),
    ],
    "performance_regression": [
        re.compile(r"OOS MAR below", re.I),
    ],
}

def classify_error_text(text: str) -> str:
    """Return category name based on regex matches."""
    for label, patterns in ERROR_PATTERNS.items():
        for pat in patterns:
            if pat.search(text or ""):
                return label
    return "unknown"
