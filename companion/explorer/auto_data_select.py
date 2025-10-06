# encoding: utf-8
import os, json, time
from datetime import datetime, timezone
import pandas as pd
import mlflow

SEED=777
def _utc_ts(): return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def load_health_cache(path):
    # Expect JSON: { "metric_slug": {"status":"ok|warn|bad", "updated_at":"ISO8601"} }
    if not path or not os.path.exists(path): return {}
    with open(path,"r",encoding="utf-8") as f: return json.load(f)

def choose_subset(ranks_csv, health_json, K=5, staleness_days=365):
    ranks = pd.read_csv(ranks_csv) if os.path.exists(ranks_csv) else pd.DataFrame(columns=["symbol","timeframe","feature","mean_importance","runs","rank"])
    health = load_health_cache(health_json)
    out = {}
    now = pd.Timestamp.utcnow()
    for (sym, tf), df in ranks.groupby(["symbol","timeframe"]):
        chosen=[]
        for _,row in df.sort_values(["rank","mean_importance"], ascending=[True,False]).iterrows():
            feat = str(row["feature"])
            h = health.get(feat, {"status":"ok","updated_at": now.isoformat()})
            status = h.get("status","ok")
            updated = pd.to_datetime(h.get("updated_at", now.isoformat()), utc=True, errors="coerce")
            if status=="bad": continue
            if (now - updated).days > staleness_days: continue
            chosen.append(feat)
            if len(chosen)>=K: break
        out[(sym,tf)] = chosen
    return out

def emit_selection(strategy, ranks_csv, health_json, K=5, staleness_days=365, out_root="artifacts/auto_data", mlflow_tracking_uri=None):
    sel = choose_subset(ranks_csv, health_json, K=K, staleness_days=staleness_days)
    ts = _utc_ts()
    out_dir = os.path.join(out_root, ts)
    os.makedirs(out_dir, exist_ok=True)
    mapping = [{"strategy":strategy,"symbol":k[0],"timeframe":k[1],"metrics":v} for k,v in sel.items()]
    out_json = os.path.join(out_dir, "selection.json")
    with open(out_json,"w",encoding="utf-8") as f: json.dump(mapping, f, ensure_ascii=False, indent=2)
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("auto_data_selection")
    with mlflow.start_run(run_name=f"auto_data_{strategy}_{ts}"):
        mlflow.set_tags({"phase":"11","module":"auto_data"})
        mlflow.log_artifact(out_json, artifact_path="auto_data")
        mlflow.log_metric("pairs", len(mapping))
    return {"out_dir": out_dir, "json": out_json}
