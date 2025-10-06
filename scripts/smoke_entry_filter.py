# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, os
from pathlib import Path
import pandas as pd
import mlflow

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Force local file tracking & a named experiment (no registry) ---
# Use a file:// URI (Windows-safe) and make sure registry is unset for this process.
tracking_dir = (REPO_ROOT / "mlruns").resolve()
tracking_uri = "file:///" + str(tracking_dir).replace("\\", "/")  # e.g. file:///C:/Users/...

# Clear any registry URI to avoid "unsupported registry URI" errors with local file store.
os.environ.pop("MLFLOW_REGISTRY_URI", None)

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("trading-lab")

from companion.ml.entry_filter import filter_with_model


# toy entries with timestamp index
idx = pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03"], utc=True)
entries = pd.DataFrame({"side": ["buy","buy","sell"]}, index=idx)

# toy features aligned to entries
X = pd.DataFrame({"f1":[0.1, 0.9, 0.6], "f2":[0.2, 0.8, 0.3]}, index=idx)

# tiny model that returns fixed probs via predict_proba
class TinyModel:
    def predict_proba(self, X):
        import numpy as np
        # make prob ~ average of columns
        p = X.mean(axis=1).to_numpy()
        p = p.reshape(-1,1)
        return np.hstack([1-p, p])

mdl = TinyModel()

with mlflow.start_run(run_name="SMOKE: entry_filter"):
    out = filter_with_model(model=mdl, X=X, entries=entries, threshold=0.6)
    kept = out["stats"]["kept"]
    total = out["stats"]["total"]
    ar = out["stats"]["accept_rate"]
    print("Kept/Total/AR:", kept, total, ar)

    mlflow.log_metric("filter_kept", float(kept))
    mlflow.log_metric("filter_total", float(total))
    mlflow.log_metric("filter_accept_rate", float(ar))
