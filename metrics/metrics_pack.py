# -*- coding: utf-8 -*-
\"\"\"
Phase 5 core metrics pack (unchanged) + Phase 6 helper for MLflow file logging.
Only the helper is new and safe to import from main env.
\"\"\"

from __future__ import annotations
import os
from typing import Dict, Any, Optional

def ensure_mlflow_logged_files(run_key: str, cfg: Optional[Dict[str, Any]], files: Dict[str, str]) -> None:
    \"\"\"Optional: log error files to MLflow if enabled.
    This is deliberately tiny to avoid entangling metrics logic.
    \"\"\"
    if not cfg or not cfg.get("enabled"):
        return
    try:
        import mlflow  # type: ignore
    except Exception:
        return
    try:
        if cfg.get("tracking_uri"):
            mlflow.set_tracking_uri(cfg["tracking_uri"])
        run_id = cfg.get("run_id")
        if run_id:
            with mlflow.start_run(run_id=run_id):
                for _, path in files.items():
                    if os.path.exists(path):
                        mlflow.log_artifact(path, artifact_path="errors")
        else:
            with mlflow.start_run(run_name=f"failed/{run_key}"):
                for _, path in files.items():
                    if os.path.exists(path):
                        mlflow.log_artifact(path, artifact_path="errors")
    except Exception:
        pass