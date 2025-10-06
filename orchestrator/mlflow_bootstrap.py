# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path

def init_mlflow(tracking_dir: str | None = None, experiment: str = "trading-lab") -> str:
    """
    Default to file:///.../mlruns, clear registry URI, set experiment.
    Safe to call multiple times.
    """
    import os, mlflow
    repo_root = Path(__file__).resolve().parents[1]
    td = Path(tracking_dir) if tracking_dir else (repo_root / "mlruns")
    td.mkdir(parents=True, exist_ok=True)

    # Use a file:// URI (Windows-safe)
    tracking_uri = "file:///" + str(td.resolve()).replace("\\", "/")
    # Ensure registry is off for local file store
    os.environ.pop("MLFLOW_REGISTRY_URI", None)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return tracking_uri