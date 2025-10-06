from .mlflow_compat import *  # applies MLflow client compat on import
# --- MLflow compatibility shim: ensure MlflowClient.list_experiments exists ---
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    if not hasattr(MlflowClient, "list_experiments"):
        def _list_experiments(self, view_type=None, max_results=None, filter_string=None, page_token=None):
            # Delegate to search_experiments for older MLflow versions
            return self.search_experiments(
                view_type=view_type,
                max_results=max_results,
                filter_string=filter_string,
                page_token=page_token,
            )
        MlflowClient.list_experiments = _list_experiments  # type: ignore[attr-defined]
except Exception:
    pass
# ---------------------------------------------------------------------------
# encoding: utf-8
import os, glob, random
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 777
random.seed(SEED); np.random.seed(SEED)

def _utc_ts():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def _infer_sym_tf(path_or_row):
    # Try columns first (when called with a dict)
    if isinstance(path_or_row, dict):
        sym = path_or_row.get("symbol")
        tf  = path_or_row.get("timeframe")
        if sym and tf:
            return str(sym), str(tf)
    # Else infer from parent folder like .../BTCUSD@15m/feature_importances.csv
    s = os.path.basename(os.path.dirname(str(path_or_row)))
    if "@" in s:
        parts = s.split("@")
        if len(parts) == 2:
            return parts[0], parts[1]
    return "UNKNOWN", "UNKNOWN"

def collect_feature_importances(search_roots):
    rows = []
    for root in search_roots:
        for f in glob.glob(os.path.join(root, "**", "feature_importances.csv"), recursive=True):
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if "feature" in df.columns and "importance" in df.columns:
                for rec in df.to_dict("records"):
                    sym = rec.get("symbol")
                    tf  = rec.get("timeframe")
                    if not sym or not tf:
                        sym, tf = _infer_sym_tf(f)
                    try:
                        rows.append({
                            "symbol": str(sym),
                            "timeframe": str(tf),
                            "feature": str(rec["feature"]),
                            "importance": float(rec["importance"]),
                        })
                    except Exception:
                        continue
    if not rows:
        return pd.DataFrame(columns=["symbol","timeframe","feature","importance"])
    return pd.DataFrame(rows)

def rank_importances(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(columns=["symbol","timeframe","feature","mean_importance","runs","rank"])
    g = df.groupby(["symbol","timeframe","feature"]).agg(
        mean_importance=("importance","mean"),
        runs=("importance","count")
    ).reset_index()
    g["rank"] = g.groupby(["symbol","timeframe"])["mean_importance"].rank(ascending=False, method="dense").astype(int)
    return g.sort_values(["symbol","timeframe","rank","feature"]).reset_index(drop=True)

def render_heatmap(ranked: pd.DataFrame, out_png: str):
    if ranked.empty:
        plt.figure(figsize=(4,2))
        plt.text(0.5,0.5,"No data", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_png, bbox_inches="tight"); plt.close(); return
    ranked = ranked.copy()
    ranked["symtf"] = ranked["symbol"].astype(str) + "@" + ranked["timeframe"].astype(str)
    piv = ranked.pivot_table(index="feature", columns="symtf", values="mean_importance", aggfunc="mean").fillna(0.0)
    plt.figure(figsize=(max(6, 0.25*len(piv.columns) + 2), max(6, 0.2*len(piv.index) + 2)))
    plt.imshow(piv.values, aspect="auto")
    plt.xticks(range(piv.shape[1]), list(piv.columns), rotation=45, ha="right", fontsize=8)
    plt.yticks(range(piv.shape[0]), list(piv.index), fontsize=8)
    plt.colorbar()
    plt.title("Feature Importance Heatmap")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight"); plt.close()

def run_tracker(search_roots, mlflow_tracking_uri=None, output_root="artifacts/feature_rank"):
    ts = _utc_ts()
    out_dir = os.path.join(output_root, ts)
    os.makedirs(out_dir, exist_ok=True)

    df = collect_feature_importances(search_roots)
    ranked = rank_importances(df)

    out_csv = os.path.join(out_dir, "importance_rank.csv")
    ranked.to_csv(out_csv, index=False, encoding="utf-8")

    out_png = os.path.join(out_dir, "heatmap.png")
    render_heatmap(ranked, out_png)

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("feature_importance_tracker")
    with mlflow.start_run(run_name=f"feature_rank_{ts}"):
        mlflow.set_tags({"phase":"11","module":"feature_rank"})
        mlflow.log_metric("rows", int(len(df)))
        mlflow.log_metric("ranked_rows", int(len(ranked)))
        mlflow.log_artifact(out_csv, artifact_path="feature_rank")
        mlflow.log_artifact(out_png, artifact_path="feature_rank")

    return {"out_dir": out_dir, "csv": out_csv, "png": out_png, "rows": int(len(df)), "ranked_rows": int(len(ranked))}

if __name__ == "__main__":
    roots = ["artifacts", "artifacts\\_ci", "artifacts\\auto_tuner", "artifacts\\monitor"]
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    run_tracker(roots, mlflow_tracking_uri=uri)



