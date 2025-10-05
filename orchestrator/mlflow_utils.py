# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow

def start_minimal_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
    active = mlflow.active_run()
    if active is None:
        mlflow.start_run(run_name=run_name)
    if tags:
        for k, v in tags.items():
            mlflow.set_tag(k, v)
    return mlflow.active_run().info.run_id

def tag_run(tags: Dict[str, str]) -> None:
    for k, v in tags.items():
        mlflow.set_tag(k, v)

def log_json_artifact(obj_or_path, artifact_path: str = "", filename: str = "health_summary.json") -> str:
    if isinstance(obj_or_path, (str, os.PathLike)) and Path(obj_or_path).exists():
        mlflow.log_artifact(str(obj_or_path), artifact_path=artifact_path or None)
        return str(obj_or_path)
    out_dir = Path(".mlflow_tmp_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / filename
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(obj_or_path, f, ensure_ascii=False, indent=2)
    mlflow.log_artifact(str(fpath), artifact_path=artifact_path or None)
    return str(fpath)

def mark_skipped_health() -> None:
    mlflow.set_tag("run_status", "SKIPPED")
    mlflow.set_tag("skip_reason", "health_check")