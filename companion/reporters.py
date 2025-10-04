"""
Reporting utilities for the AI companion.

This module provides helper functions to log decisions and outcomes
both to local log files and to MLflow as tags or notes.  Maintaining
a clear audit trail is important for reproducibility and compliance.

Logs are written under ``companion/logs``.  MLflow tags are written
using the MLflow client; this module does not depend on the rest of
the companion to avoid circular imports.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import mlflow  # type: ignore
except ImportError:
    mlflow = None

LOG_DIR = Path("companion/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "companion.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def log_decision(plan_id: str, decision: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Record a decision made by the AI companion.

    Parameters
    ----------
    plan_id: str
        Identifier of the patch plan to which this decision relates.
    decision: str
        A brief description of the action taken (e.g. "applied",
        "rejected", "reran").
    details: Optional[Dict[str, Any]]
        Additional structured information about the decision.  This
        dictionary will be serialised to JSON and appended to the log.
    """
    timestamp = datetime.utcnow().isoformat()
    record = {
        "timestamp": timestamp,
        "plan_id": plan_id,
        "decision": decision,
        "details": details or {},
    }
    logging.info(json.dumps(record))

def tag_run(run_id: str, key: str, value: str) -> None:
    """Add or update a tag on an MLflow run.

    Parameters
    ----------
    run_id: str
        The MLflow run identifier.
    key: str
        Tag name.
    value: str
        Tag value.
    """
    if mlflow is None:
        raise RuntimeError("mlflow is not installed. Install mlflow to tag runs.")
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, key, value)

__all__ = ["log_decision", "tag_run"]
