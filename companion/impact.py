"""
Impact analysis for patch plans.

After a patch plan is generated, we need to determine which studies or
jobs are affected by the proposed changes so that only those are
re‑run.  This module encapsulates that logic.  Given a plan and
optionally the context of the run being patched, it returns a list of
identifiers (strategy, symbol, timeframe) that need to be re‑executed.

The default implementation returns an empty list; concrete logic
should be implemented based on the specifics of your project (e.g.
mapping strategy file changes to certain strategy names).
"""

from __future__ import annotations

from typing import Any, Dict, List

def get_impacted_jobs(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compute the list of impacted studies/jobs from a patch plan.

    Parameters
    ----------
    plan: Dict[str, Any]
        The structured plan (as a dictionary) describing proposed
        changes.  It may contain information about which files or
        parameters are modified.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries with keys such as ``strategy``, ``symbol``,
        and ``timeframe``.  Use the project’s own naming conventions.
    """
    # TODO: implement a real impact analysis based on file names and
    # configuration changes.  For now we return an empty list.
    return []

__all__ = ["get_impacted_jobs"]
