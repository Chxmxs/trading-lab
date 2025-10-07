# -*- coding: utf-8 -*-
"""
Export Optuna/MLflow trial history to artifacts/optuna/trials.parquet
- Real MLflow store only (file:./mlruns)
- Exports runs that have at least ONE tuned param and ONE objective-like metric
- Accepts params with 'param_' OR 'param.'; normalizes to 'param_' in output
- Accepts metrics with dots or underscores; tries smart fallbacks
- Deterministic; PS5-safe; no installs

CLI:
  python scripts/export_optuna_history.py --objective-keys objective.final,oos.mar,oos.erpt,oos.sortino,oos.sharpe
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import mlflow

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "optuna"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "trials.parquet"

def _ms_to_iso(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None

def parse_args():
    ap = argparse.ArgumentParser(description="Export MLflow runs to trials.parquet for the tuner.")
    ap.add_argument(
        "--objective-keys",
        default="objective.final,oos.mar,oos.erpt,oos.sortino,oos.sharpe,oos_mar,objective,mar,expected_return_per_trade,erpt,oos_sortino,oos_sharpe",
        help="Comma-separated metric names in priority order (dots/underscores both allowed)."
    )
    ap.add_argument("--min-runs", type=int, default=1, help="Require at least this many qualifying runs overall.")
    ap.add_argument("--max-per-exp", type=int, default=5000, help="Max runs per experiment to scan.")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()

def _norm_key(k: str) -> str:
    return k.strip().lower()

def _alts(name: str) -> List[str]:
    """Generate alternate spellings for a metric name (dot/underscore variants)."""
    n = _norm_key(name)
    alts = {n, n.replace(".", "_"), n.replace("_", ".")}
    # specific common prefixes
    if n.startswith("oos."):
        alts.add(n[4:])            # oos.mar -> mar
        alts.add("oos_" + n[4:])   # oos_...
    if n.startswith("oos_"):
        alts.add(n[4:])
        alts.add("oos." + n[4:])
    if n.startswith("objective."):
        alts.add(n.replace("objective.", "objective_"))
        alts.add("objective")
    if n == "objective":
        alts.add("objective.final")
    return list(alts)

def _pick_objective(metrics: Dict[str, float], candidates: List[str]) -> Optional[float]:
    md = { _norm_key(k): v for k, v in metrics.items() }
    # 1) direct match across dot/underscore alternates
    for c in candidates:
        for a in _alts(c):
            if a in md:
                return float(md[a])
    # 2) fuzzy: prefer any oos.* metrics that look like performance
    for k in md:
        if k.startswith("oos.") or k.startswith("oos_"):
            if any(s in k for s in ("mar","sharpe","sortino","erpt","return","profit","cagr")):
                return float(md[k])
    # 3) last resort: objective.* (e.g., objective.final)
    for k in md:
        if k.startswith("objective.") or k.startswith("objective_"):
            return float(md[k])
    return None

def _collect_params(raw_params: Dict[str, str]) -> Dict[str, object]:
    """Accept both param_... and param.... Return dict with 'param_' keys."""
    out: Dict[str, object] = {}
    for k, v in raw_params.items():
        lk = _norm_key(k)
        if lk.startswith("param_"):
            name = lk  # already normalized
        elif lk.startswith("param."):
            name = "param_" + lk.split("param.", 1)[1].replace(".", "_")
        else:
            continue
        # cast to int/float where possible, else keep str
        try:
            sv = str(v).strip()
            if sv.isdigit() or (sv.startswith("-") and sv[1:].isdigit()):
                out[name] = int(sv)
            else:
                out[name] = float(sv)
        except Exception:
            out[name] = str(v)
    return out

def main() -> int:
    args = parse_args()
    CANDIDATES = [_norm_key(x) for x in args.objective_keys.split(",") if x.strip()]

    mlflow.set_tracking_uri("file:./mlruns")
    client = mlflow.tracking.MlflowClient()

    try:
        experiments = client.search_experiments()
    except Exception as e:
        print(f"[EXPORT] Could not enumerate experiments: {e}")
        return 2

    rows = []
    for exp in experiments:
        try:
            runs = client.search_runs(
                [exp.experiment_id],
                max_results=args.max_per_exp,
                order_by=["attributes.end_time DESC"]
            )
        except Exception as e:
            print(f"[EXPORT] Failed to search runs for experiment {exp.name}: {e}")
            continue

        for r in runs:
            params = { _norm_key(k): v for k, v in r.data.params.items() }
            metrics = { _norm_key(k): v for k, v in r.data.metrics.items() }
            tags = { _norm_key(k): v for k, v in r.data.tags.items() }

            param_cols = _collect_params(params)
            if not param_cols:
                # no tunable params → skip
                continue

            obj = _pick_objective(metrics, CANDIDATES)
            if obj is None:
                # no objective metric we recognize → skip
                continue

            strategy = params.get("strategy") or tags.get("strategy") or params.get("strategy_name") or "unknown"
            symbol = params.get("symbol") or tags.get("symbol") or "unknown"
            timeframe = params.get("timeframe") or tags.get("timeframe") or "unknown"

            trial_number = None
            for k in ("trial_number","optuna_trial_number","trial","optuna_trial"):
                if k in params:
                    try: trial_number = int(params[k]); break
                    except Exception: pass
                if k in tags:
                    try: trial_number = int(tags[k]); break
                    except Exception: pass

            row = {
                "strategy": str(strategy),
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "trial_number": int(trial_number) if trial_number is not None else -1,
                "run_id": r.info.run_id,
                "exp": exp.name,
                "end_time": _ms_to_iso(r.info.end_time),
                "objective": float(obj),
            }
            # retain oos_mar alias if present, else mirror objective
            row["oos_mar"] = float(metrics.get("oos.mar", metrics.get("oos_mar", obj)))

            row.update(param_cols)
            rows.append(row)

    if not rows:
        print("[EXPORT] No qualifying runs found in MLflow (no objective metric from your list or no tuned params).")
        return 2

    df = pd.DataFrame(rows)
    if "end_time" in df.columns:
        df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
    df = df.sort_values(["end_time", "trial_number"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    print(f"[EXPORT] Wrote {OUT_FILE} with {len(df)} rows and {df.shape[1]} columns.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
