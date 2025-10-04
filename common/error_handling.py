# -*- coding: utf-8 -*-
"""
Phase 6: Centralized error capture, backoff, replay bundles, and quarantine.
Main env only (tradebot). No mlfinlab imports here.
All timestamps in UTC. Windows-safe path parts.
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
import zipfile
import random
import traceback
import datetime as _dt
from typing import Any, Callable, Dict, Optional

# ---------- Time & FS helpers ----------

def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _write_text(path: str, text: str) -> None:
    _safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def _append_text(path: str, text: str) -> None:
    _safe_mkdir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)

# FS name sanitizer (Windows-safe)
def _sanitize_for_fs(s: Any) -> str:
    try:
        s = str(s)
    except Exception:
        s = "unknown"
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in s)
    out = out.replace(" ", "_").strip().strip(".")
    return out[:180]  # keep paths short-ish

def ensure_boot_artifact_dir(base_artifacts: str, key: str, run_id: str) -> str:
    key = _sanitize_for_fs(key)
    run_id = _sanitize_for_fs(run_id)
    outdir = os.path.join(base_artifacts, "_boot", key, run_id)
    _safe_mkdir(outdir)
    return outdir

# ---------- Exception classification ----------

def is_transient(exc: BaseException) -> bool:
    """Heuristics for classifying transient/retryable errors."""
    transient_types = (
        TimeoutError,
        FileExistsError,
        BrokenPipeError,
        ConnectionError,
    )
    if isinstance(exc, transient_types):
        return True

    if isinstance(exc, OSError):
        if getattr(exc, "errno", None) == 24:  # too many open files
            return True

    msg = (str(exc) or "").lower()
    patterns = [
        "brokenprocesspool",
        "remoteerror",
        "ray::",
        "rayinitializationerror",
        "object store is full",
        "file is being used by another process",
        "permission denied",
        "resource temporarily unavailable",
        "lock",
    ]
    return any(p in msg for p in patterns)

# ---------- Backoff & retry ----------

def with_backoff(
    fn: Callable[[], Any],
    *,
    max_retries: int = 2,
    base: float = 0.5,
    cap: float = 5.0,
    jitter: bool = True,
    on_retry_log: Optional[Callable[[int, float, BaseException], None]] = None,
) -> Any:
    attempt = 0
    while True:
        try:
            return fn()
        except BaseException as e:
            if is_transient(e) and attempt < max_retries:
                delay = min(cap, base * (2 ** attempt))
                if jitter:
                    delay = delay * (0.5 + random.random())
                attempt += 1
                if on_retry_log:
                    on_retry_log(attempt, delay, e)
                time.sleep(delay)
                continue
            raise

# ---------- Capture exception artifacts ----------

def _exc_summary(e: BaseException) -> str:
    return f"{e.__class__.__name__}: {e}"

def capture_exception(outdir: str, e: BaseException) -> Dict[str, str]:
    _safe_mkdir(outdir)
    exc_path = os.path.join(outdir, "exception.txt")
    tb_path = os.path.join(outdir, "traceback.txt")
    _write_text(exc_path, _exc_summary(e))
    _write_text(tb_path, "".join(traceback.format_exception(type(e), e, e.__traceback__)))
    return {"exception": exc_path, "traceback": tb_path}

# ---------- Replay bundle ----------

def _slice_df(df, max_rows: int = 200):
    try:
        import pandas as pd  # local import
    except Exception:
        return None
    if df is None:
        return None
    try:
        if len(df) <= max_rows:
            return df
        head_n = min(25, max_rows // 2)
        tail_n = max_rows - head_n
        return pd.concat([df.head(head_n), df.tail(tail_n)], axis=0)
    except Exception:
        return None

def _safe_to_csv(path: str, df) -> Optional[str]:
    if df is None:
        return None
    try:
        df.to_csv(path, index=True)
        return path
    except Exception:
        return None

def write_replay_bundle(outdir: str, context: Dict[str, Any]) -> str:
    _safe_mkdir(outdir)
    bundle_path = os.path.join(outdir, "replay_bundle.zip")
    staging = os.path.join(outdir, "_replay_stage")
    if os.path.exists(staging):
        shutil.rmtree(staging, ignore_errors=True)
    _safe_mkdir(staging)

    params = {
        "strategy": context.get("strategy"),
        "symbol": context.get("symbol"),
        "timeframe": context.get("timeframe"),
        "params": context.get("params"),
        "run_config": context.get("run_config"),
        "key": context.get("key"),
        "cv_dir": context.get("cv_dir"),
        "environment": {
            "env_name": context.get("env_name", "tradebot"),
            "python": sys.version,
            "executable": sys.executable,
            "cwd": os.getcwd(),
        },
        "timestamps": {"created_utc": _utc_now_iso()},
    }
    _write_text(os.path.join(staging, "params.json"), json.dumps(params, indent=2, sort_keys=True))

    df = context.get("df")
    df_slice = _slice_df(df, max_rows=200)
    if df_slice is not None:
        _safe_to_csv(os.path.join(staging, "df_sample.csv"), df_slice)

    events = context.get("events")
    events_slice = _slice_df(events, max_rows=200)
    if events_slice is not None:
        _safe_to_csv(os.path.join(staging, "events_sample.csv"), events_slice)

    cv_dir = context.get("cv_dir")
    folds_meta = {}
    if isinstance(cv_dir, str) and os.path.isdir(cv_dir):
        folds_meta = {"label_store_dir": cv_dir}
        folds_json = os.path.join(cv_dir, "folds.json")
        if os.path.exists(folds_json):
            try:
                with open(folds_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    folds_meta["first_fold"] = data[0]
            except Exception:
                pass
    _write_text(os.path.join(staging, "folds_meta.json"), json.dumps(folds_meta, indent=2, sort_keys=True))

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(staging):
            for fn in files:
                fp = os.path.join(root, fn)
                zf.write(fp, os.path.relpath(fp, staging))

    shutil.rmtree(staging, ignore_errors=True)
    return bundle_path

# ---------- Quarantine ----------

def quarantine(
    outdir: str,
    key: str,
    reason: str,
    e: BaseException,
    context: Dict[str, Any],
) -> str:
    base_artifacts = context.get("artifacts_root") or os.path.join("artifacts")
    run_id = os.path.basename(outdir.rstrip(os.sep))
    qdir = os.path.join(base_artifacts, "_quarantine", _sanitize_for_fs(key), _sanitize_for_fs(run_id))
    _safe_mkdir(os.path.dirname(qdir))
    try:
        if os.path.abspath(outdir) != os.path.abspath(qdir):
            if os.path.exists(qdir):
                shutil.rmtree(qdir, ignore_errors=True)
            shutil.move(outdir, qdir)
    except Exception:
        shutil.copytree(outdir, qdir, dirs_exist_ok=True)

    qmeta = {
        "reason": reason,
        "exception_type": type(e).__name__,
        "exception_message": str(e),
        "job": {
            "strategy": context.get("strategy"),
            "symbol": context.get("symbol"),
            "timeframe": context.get("timeframe"),
            "params": context.get("params"),
        },
        "paths": {
            "artifact_dir": outdir,
            "quarantine_dir": qdir,
            "cv_dir": context.get("cv_dir"),
        },
        "timestamps": {"utc": _utc_now_iso()},
        "key": key,
    }
    _write_text(os.path.join(qdir, "quarantine.json"), json.dumps(qmeta, indent=2, sort_keys=True))
    print(f"[QUARANTINE] {key} -> {qdir}")
    return qdir

# ---------- MLflow helper (optional) ----------

def _maybe_log_artifacts_to_mlflow(mlflow_cfg: Optional[Dict[str, Any]], files: Dict[str, str]) -> None:
    if not mlflow_cfg or not mlflow_cfg.get("enabled"):
        return
    try:
        import mlflow  # type: ignore
    except Exception:
        return
    try:
        if mlflow_cfg.get("tracking_uri"):
            mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        run_id = mlflow_cfg.get("run_id")
        if run_id:
            with mlflow.start_run(run_id=run_id):
                for _, path in files.items():
                    if os.path.exists(path):
                        mlflow.log_artifact(path, artifact_path="errors")
        else:
            key = mlflow_cfg.get("key", "unknown")
            with mlflow.start_run(run_name=f"failed/{key}"):
                for _, path in files.items():
                    if os.path.exists(path):
                        mlflow.log_artifact(path, artifact_path="errors")
    except Exception:
        pass

# ---------- Wrap & execute with capture ----------

def wrap_with_error_capture(
    func: Callable[[], Any],
    *,
    context: Dict[str, Any],
    mlflow_cfg: Optional[Dict[str, Any]] = None,
    max_retries: int = 2,
) -> Any:
    key = _sanitize_for_fs(context.get("key", "unknown"))
    outdir = context.get("outdir")
    base_artifacts = context.get("artifacts_root") or os.path.join("artifacts")
    paramhash = _sanitize_for_fs(context.get("paramhash") or "unknown")
    timestamp = context.get("timestamp") or _utc_now_iso()
    timestamp = timestamp.replace(":", "").replace("-", "")
    timestamp = _sanitize_for_fs(timestamp)
    run_id = _sanitize_for_fs(f"{timestamp}__{paramhash}")

    if not outdir:
        outdir = ensure_boot_artifact_dir(base_artifacts, key, run_id)
        context["outdir"] = outdir

    def _on_retry(attempt: int, delay: float, exc: BaseException):
        msg = f"[RETRY {attempt}/{max_retries} in {delay:.2f}s] {_exc_summary(exc)}"
        print(msg)
        _append_text(os.path.join(outdir, "traceback.txt"), f"\n{_utc_now_iso()}  {msg}\n")

    try:
        result = with_backoff(
            func,
            max_retries=max_retries,
            on_retry_log=_on_retry,
        )
        return result
    except BaseException as e:
        files = capture_exception(outdir, e)
        try:
            bundle = write_replay_bundle(outdir, context)
            files["replay_bundle"] = bundle
        except Exception:
            pass
        _maybe_log_artifacts_to_mlflow({**(mlflow_cfg or {}), "key": key}, files)
        reason = "retries_exhausted" if is_transient(e) else "persistent"
        print(f"[FAIL] {key}: {_exc_summary(e)}")
        quarantine(outdir, key, reason, e, context)
        return None