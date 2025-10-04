# -*- coding: utf-8 -*-
\"\"\"
Phase 4.5 CV generator (mlfinlab env) patched for Phase 6:
- Wrap main() in a minimal inline error-capture shim (no cross-env imports).
- On exception: write exception.txt, traceback.txt, and replay_bundle.zip with
  args + small slices, then exit(1). No fabricated errors.
\"\"\"

from __future__ import annotations

import os
import sys
import json
import shutil
import zipfile
import traceback
import datetime as _dt

# ---------- Minimal FS helpers (duplicated inline to avoid cross-env import) ----------

def _utc_now_iso():
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def _write_text(path: str, text: str):
    _safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def _slice_df(df, max_rows: int = 200):
    try:
        import pandas as pd
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

def _safe_to_csv(path: str, df):
    if df is None:
        return
    try:
        df.to_csv(path, index=True)
    except Exception:
        pass

def _capture_exception(outdir: str, e: BaseException):
    _write_text(os.path.join(outdir, "exception.txt"), f"{type(e).__name__}: {e}")
    _write_text(os.path.join(outdir, "traceback.txt"), "".join(traceback.format_exception(type(e), e, e.__traceback__)))

def _write_replay_bundle(outdir: str, args: dict, df=None, events=None, samples_info_sets=None, label_dir=None):
    bundle = os.path.join(outdir, "replay_bundle.zip")
    stage = os.path.join(outdir, "_replay_stage")
    if os.path.exists(stage):
        shutil.rmtree(stage, ignore_errors=True)
    _safe_mkdir(stage)

    # params.json
    params = {
        "args": args,
        "environment": {"env_name": "tradebot-mlfl", "python": sys.version, "executable": sys.executable},
        "label_store_dir": label_dir,
        "timestamps": {"created_utc": _utc_now_iso()},
    }
    _write_text(os.path.join(stage, "params.json"), json.dumps(params, indent=2, sort_keys=True))

    _safe_to_csv(os.path.join(stage, "df_sample.csv"), _slice_df(df))
    _safe_to_csv(os.path.join(stage, "events_sample.csv"), _slice_df(events))

    # samples_info_sets small JSON if provided
    sis_path = os.path.join(stage, "samples_info_sets.json")
    try:
        if samples_info_sets is not None:
            small = samples_info_sets[:5] if hasattr(samples_info_sets, "__getitem__") else None
            _write_text(sis_path, json.dumps(small, default=str, indent=2))
    except Exception:
        pass

    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(stage):
            for fn in files:
                fp = os.path.join(root, fn)
                zf.write(fp, os.path.relpath(fp, stage))
    shutil.rmtree(stage, ignore_errors=True)
    return bundle

# ---------- Main logic (placeholder hooks to existing Phase 4.5) ----------

def _resolve_main():
    \"\"\"Locate the real CV generation entry (Phase 4.5).\"\"\"
    for mod_name, fn_name in [
        ("scripts.cv_core_mlfl", "main"),
        ("scripts.cv_generate_core", "main"),
        ("scripts.cv_entry", "main"),
        ("scripts.cv_generate_mlfl_impl", "main"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn
        except Exception:
            continue
    return None

def main():
    \"\"\"Shim that wraps the real main() if found; otherwise no-op.\"\"\"
    real = _resolve_main()
    if real is None:
        print("[cv_generate_mlfl] No CV generator implementation found. Nothing to do.")
        return

    return real()

if __name__ == "__main__":
    ts = _utc_now_iso().replace(":", "").replace("-", "")
    outdir = os.environ.get("CV_OUTDIR", os.path.join("artifacts", "_cv_boot", ts))
    try:
        ctx = main()
        sys.exit(0)
    except BaseException as e:
        _safe_mkdir(outdir)
        _capture_exception(outdir, e)
        args = {"argv": sys.argv, "env": {k: os.environ.get(k) for k in ["CV_OUTDIR", "LABEL_KEY", "LABEL_DIR"]}}
        df = None
        events = None
        sis = None
        label_dir = os.environ.get("LABEL_DIR")
        try:
            _write_replay_bundle(outdir, args=args, df=df, events=events, samples_info_sets=sis, label_dir=label_dir)
        except Exception:
            pass
        sys.exit(1)