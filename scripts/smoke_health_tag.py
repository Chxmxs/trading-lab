# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
from pathlib import Path

# Ensure repo import path regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlflow
from mlflow.tracking import MlflowClient
from orchestrator.execute import run_one_core, _maybe_tag_active_mlflow_with_health
import companion.data_checks.health as health_mod

# --- Force WARN so the run proceeds (not SKIPPED) -----------------------------
def fake_sum(**kws):
    return {"status": "warn", "warnings": ["demo warning"], "errors": []}
health_mod.summarize_health = fake_sum

# --- Pin tracking URI (use env from shell: file:///.../artifacts/mlruns) -----
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not tracking_uri:
    tracking_path = REPO_ROOT / "artifacts" / "mlruns"
    tracking_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = "file:///" + str(tracking_path).replace("\\", "/")
mlflow.set_tracking_uri(tracking_uri)

# --- Create/get experiment (ensure nonzero id), then create run via client ----
EXP_NAME = "trading-lab"
client = MlflowClient()
exp = client.get_experiment_by_name(EXP_NAME)
if exp is None:
    exp_id = client.create_experiment(EXP_NAME)
else:
    exp_id = exp.experiment_id

print("Tracking URI:", mlflow.get_tracking_uri())
print("Experiment ID (nonzero expected):", exp_id)

# --- Minimal context to exercise the health preflight + tag/log path ----------
ci_root = REPO_ROOT / "artifacts" / "_ci" / "smoke_health"
ci_root.mkdir(parents=True, exist_ok=True)
( ci_root / "ohlcv.csv" ).write_text("ts,open,high,low,close,volume", encoding="utf-8")
( ci_root / "metric.csv").write_text("ts,value",                        encoding="utf-8")

context = {
    "symbol": "BTCUSD",
    "timeframe": "15m",
    "start_dt": "2020-01-01",
    "end_dt": "2020-01-10",
    "price_csv": str(ci_root / "ohlcv.csv"),
    "data_sources": [
        str(ci_root / "ohlcv.csv"),
        str(ci_root / "metric.csv"),
    ],
    # Minimal strategy stub that returns fake outputs
    "strategy_obj": type("S", (), {
        "run": lambda self, df, cfg: (
            __import__("pandas").Series([1, 2, 3], name="equity"),
            __import__("pandas").DataFrame({"ts": [1]})
        )
    })(),
    "df": None,
    "run_config": {},
}

# --- Create the run explicitly, then activate it by run_id --------------------
new_run = client.create_run(experiment_id=str(exp_id), run_name="SMOKE: health pass/warn tagging")
print("Created run_id:", new_run.info.run_id)

with mlflow.start_run(run_id=new_run.info.run_id):
    # run_one_core computes data health, stashes into context
    result = run_one_core(context)
    # If tagging couldn't happen earlier, do it now on the active run
    _maybe_tag_active_mlflow_with_health(context)
    print("RESULT:", result)
    mlflow.log_metric("smoke_ok", 1.0)
