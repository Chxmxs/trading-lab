"""
Failure taxonomy and classification rules for the AI companion.

This module defines a taxonomy of known failure modes encountered in the
trading‑lab project and provides functions to classify run artifacts
into one of these categories.  The classification logic is rule‑based
and testable: it relies solely on information contained in the
artifacts (e.g. ``debug.txt``, ``context.json``, equity/trades CSVs)
and does not require external state.  See :mod:`tests.test_classifier`
for examples.

The taxonomy distinguishes between data issues (e.g. missing files),
encoding problems, MLflow initialisation errors, serialization
issues, schema violations, constraint violations, parameter search
issues, and robustness tuning problems.  Each failure type maps to
a set of standard remedies (see :mod:`companion.planner`).
"""

from __future__ import annotations

import json
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd  # type: ignore

class FailureType(Enum):
    """Enumeration of known failure modes.

    Each member corresponds to a specific error class that the
    companion can recognise and handle.  The ``UNKNOWN`` type is
    reserved for cases that do not match any known rule.
    """

    DATA_MISSING = auto()        # referenced input data file not found
    ENCODING_BOM = auto()        # JSON parse failed due to BOM
    MLFLOW_INIT = auto()         # tracking URI or experiment creation error
    SERIALIZATION = auto()       # non‑serializable objects in context
    NON_DATETIME_EQUITY = auto() # equity index is not a pandas.DatetimeIndex
    NO_TRADES = auto()           # trades table missing or empty
    CONSTRAINT_MIN_TRADES = auto() # fewer trades than min_trades
    PARAM_STALLED = auto()       # objective values identical across trials
    ROBUSTNESS_OVERSHOOT = auto() # penalty dominates raw objective
    PLATEAU = auto()             # early stopper plateau detected
    UNKNOWN = auto()             # unrecognised error

def classify(artifacts_dir: Path) -> FailureType:
    """Classify the root cause of a failed or underperforming run.

    The function inspects various files in the run's artifacts
    directory to determine which failure rule applies.  Rules should
    be ordered from most specific to least specific.  If no rule
    matches, ``FailureType.UNKNOWN`` is returned.

    Parameters
    ----------
    artifacts_dir: Path
        Path to the directory containing the downloaded artifacts
        (debug.txt, context.json, equity_*.csv, trades_*.csv, etc.).

    Returns
    -------
    FailureType
        The identified failure type.
    """
    # TODO: parse debug.txt and context.json to identify BOM issues
    # and MLflow initialisation errors.  Check for missing data files
    # referenced in context.json.  Load CSVs to examine equity index
    # and trades schema.  Use pandas for simple checks.
    # This skeleton always returns UNKNOWN.
    return FailureType.UNKNOWN

__all__ = ["FailureType", "classify"]
