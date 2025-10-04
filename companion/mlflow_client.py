"""
Utilities for interacting with MLflow.

This module centralises all interactions with the MLflow tracking server
so that the rest of the AI companion does not need to know the details
of the MLflow API.  Functions here return plain Python objects and
download artifacts to the local filesystem.  When possible, the
functions should be implemented to be idempotent and resilient to
errors such as missing experiments or runs.

Example usage::

    from companion.mlflow_client import MLflowClient

    client = MLflowClient(tracking_uri="file:///path/to/mlruns")
    runs = client.fetch_runs([
        "TradingExperiment"
    ], since="3d", objective="OOS_MAR", threshold=0.2)
    for run in runs:
        art_dir = client.download_artifacts(run["run_id"], Path("tmp/artifacts"))
        # ... process artifacts ...
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import mlflow  # type: ignore
except ImportError:
    mlflow = None  # MLflow is optional in this skeleton

@dataclass
class RunInfo:
    """Simplified representation of an MLflow run used by the companion.

    Attributes
    ----------
    run_id: str
        Unique identifier of the run.
    experiment_id: str
        Identifier of the MLflow experiment.
    metrics: Dict[str, float]
        Mapping of metric keys to their latest values.
    tags: Dict[str, str]
        Mapping of tag keys to tag values.
    start_time: _dt.datetime
        Start time of the run.
    status: str
        Run status (e.g. ``FINISHED``, ``FAILED``).
    """

    run_id: str
    experiment_id: str
    metrics: Dict[str, float]
    tags: Dict[str, str]
    start_time: _dt.datetime
    status: str

class MLflowClient:
    """Thin wrapper around the MLflow tracking API.

    The constructor takes a tracking URI and initialises the underlying
    MLflow client if available.  All methods should degrade gracefully
    if MLflow is not installed by raising a clear error.
    """

    def __init__(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri
        if mlflow is None:
            raise RuntimeError("mlflow is not installed. Run `pip install mlflow` to use the mlflow_client.")
        mlflow.set_tracking_uri(tracking_uri)
        self._client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    def fetch_runs(
        self,
        experiments: Iterable[str],
        since: Optional[str] = None,
        objective: str = "OOS_MAR",
        threshold: float = 0.0,
    ) -> List[RunInfo]:
        """Query runs from the specified experiments.

        Parameters
        ----------
        experiments: Iterable[str]
            A list of experiment names or IDs to search within.
        since: Optional[str], default None
            An optional lookback period (e.g. ``"7d"``) indicating how far
            back to query.  If ``None``, all runs are considered.
        objective: str, default "OOS_MAR"
            The objective metric to evaluate.
        threshold: float, default 0.0
            Minimum acceptable value for the objective.  Runs with final
            values below this threshold are marked as underperforming.

        Returns
        -------
        List[RunInfo]
            A list of simplified run information matching the criteria.
        """
        # TODO: Parse the `since` string into a datetime cutoff and query
        # MLflow accordingly.  This skeleton returns an empty list.
        return []

    def download_artifacts(self, run_id: str, output_dir: Path) -> Path:
        """Download all artifacts for a given run into a local directory.

        The MLflow client will create the destination directory if it
        does not exist.  Existing files will be overwritten.  The
        downloaded directory path is returned to the caller.

        Parameters
        ----------
        run_id: str
            MLflow run ID whose artifacts should be downloaded.
        output_dir: Path
            Directory into which artifacts are downloaded.

        Returns
        -------
        Path
            Path to the directory containing the downloaded artifacts.
        """
        # TODO: implement using mlflow.artifacts.download_artifacts
        raise NotImplementedError("download_artifacts is a stub; implement in Phase 8")

__all__ = ["MLflowClient", "RunInfo"]
