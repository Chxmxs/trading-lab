"""
Patch planning based on classified failures.

This module maps failure types identified by the classifier to
concrete patch plans.  A patch plan is a structured dictionary
describing proposed changes to configuration files, search spaces,
or strategy adapter code.  Each plan includes metadata such as the
risk tier (A, B, or C), rationale, and a rollback plan.  Plans are
designed to be serialisable to JSON and stored under the ``patches``
directory for review.

The logic here should be deterministic given the inputs and should
avoid any network calls.  Interaction with LLMs takes place in
``ai_core.py``; this module orchestrates the output of the model
with the companion's own policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .classifier import FailureType

@dataclass
class PatchAction:
    """Represents a single change in a patch plan.

    Attributes
    ----------
    target: str
        Path or identifier of the file to change (e.g. ``"configs/tuning.json"``).
    diff: str
        Unified diff representing the proposed change.
    risk: str
        Risk tier (``"A"``, ``"B"`` or ``"C"``).
    rationale: str
        Human‑readable explanation of why this change is proposed.
    rollback: str
        Instructions on how to revert the change if necessary.
    """

    target: str
    diff: str
    risk: str
    rationale: str
    rollback: str

@dataclass
class PatchPlan:
    """Encapsulates all actions needed to resolve a failure.

    Attributes
    ----------
    plan_id: str
        Unique identifier for this plan (e.g. ``"PLAN_20250101_1200"``).
    failure: FailureType
        The failure type that prompted the plan.
    actions: List[PatchAction]
        List of proposed actions in the plan.
    impacted: List[Dict[str, Any]]
        List of impacted studies/jobs (strategy, symbol, timeframe)
        computed by :mod:`companion.impact`.  Filled later.
    prompt: str
        The exact prompt sent to the LLM when generating the plan.
        Stored for reproducibility (secrets redacted).
    model_response: Optional[Dict[str, Any]]
        Raw model response, if any.
    """

    plan_id: str
    failure: FailureType
    actions: List[PatchAction] = field(default_factory=list)
    impacted: List[Dict[str, Any]] = field(default_factory=list)
    prompt: str = ""
    model_response: Optional[Dict[str, Any]] = None

def make_plan(
    failure: FailureType,
    artifacts: Dict[str, Any],
    config: Dict[str, Any],
    ai_client: Optional[Any] = None,
) -> PatchPlan:
    """Create a patch plan based on the failure type and run artifacts.

    This function encapsulates the policy logic for each failure type.
    For simple cases (Tier A), it may generate patches directly
    without consulting the LLM.  For more complex scenarios (Tier B/C),
    it may call the supplied ``ai_client`` to solicit suggestions from
    the language model.  The prompt used and model response should be
    saved in the plan for reproducibility.

    Parameters
    ----------
    failure: FailureType
        The identified failure type.
    artifacts: Dict[str, Any]
        Parsed run artifacts (e.g. context, equity/trades data).  The
        structure of this dictionary is defined in :mod:`mlflow_client`.
    config: Dict[str, Any]
        Companion configuration loaded from ``configs/companion.json``.
    ai_client: Optional[Any], optional
        If provided, an instance of :class:`companion.ai_core.AIClient`
        used to query the LLM for suggestions.

    Returns
    -------
    PatchPlan
        A structured plan containing zero or more actions.
    """
    # TODO: implement policies per failure type.  For example:
    # if failure == FailureType.DATA_MISSING:
    #     ... build a config patch to provide the missing path ...
    # For now we return an empty plan.
    plan = PatchPlan(
        plan_id="PLAN_00000000_0000",
        failure=failure,
        actions=[],
        prompt="",  # This should be the actual prompt sent to the LLM
    )
    return plan

__all__ = ["PatchPlan", "PatchAction", "make_plan"]
