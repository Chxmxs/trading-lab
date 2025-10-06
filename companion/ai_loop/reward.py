# encoding: utf-8
import os, json, statistics
from datetime import datetime, timezone
import mlflow
import pandas as pd
SEED=777

def _utc_ts(): return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def _latest_run_for_strategy(client, experiment_name, strategy):
    exp = client.get_experiment_by_name(experiment_name)
    if not exp: return None
    runs = client.search_runs(exp.experiment_id, f"tags.strategy = '{strategy}'", order_by=["attributes.start_time DESC"], max_results=50)
    return runs[0] if runs else None

def _last_n_oos_mar(client, experiment_name, strategy, n=20):
    exp = client.get_experiment_by_name(experiment_name)
    if not exp: return []
    runs = client.search_runs(exp.experiment_id, f"tags.strategy = '{strategy}'", order_by=["attributes.start_time DESC"], max_results=n+1)
    vals=[]
    for r in runs:
        v = r.data.metrics.get("oos_mar")
        if v is not None: vals.append(float(v))
    return vals

def _topk_features(import_rank_csv, k=5):
    if not import_rank_csv or not os.path.exists(import_rank_csv): return []
    df = pd.read_csv(import_rank_csv)
    df = df.sort_values(["rank","mean_importance"], ascending=[True,False])
    return df["feature"].head(k).dropna().astype(str).tolist()

def _negative_patterns():
    try:
        from companion.ai_loop import patterns
        return getattr(patterns, "NEGATIVE_PATTERNS", [])
    except Exception:
        return []

def compute_and_log_reward(strategy, experiments, topk_from_csv, k=5, trailing_n=20, mlflow_tracking_uri=None, out_root="artifacts/reward"):
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    # choose primary exp to compute reward from; prefer "strategy training runs"
    primary = None
    for name in ["strategy training runs","adaptive_tuner","entry_filter"]:
        if name in experiments: primary = name; break
    if not primary and experiments: primary = experiments[0]
    if not primary: raise RuntimeError("No experiments supplied")

    vals = _last_n_oos_mar(client, primary, strategy, n=trailing_n+1)
    if len(vals) < 2:  # need at least 2
        latest, trailing = None, []
    else:
        latest, trailing = vals[0], vals[1:1+trailing_n]
    trailing_med = statistics.median(trailing) if trailing else 0.0
    if trailing_med == 0 or latest is None:
        reward = 0.0
    else:
        reward = (latest - trailing_med) / abs(trailing_med) * 100.0

    ts = _utc_ts()
    out_dir = os.path.join(out_root, ts)
    os.makedirs(out_dir, exist_ok=True)
    prompt = {
        "strategy": strategy,
        "reward_scalar": round(reward, 6),
        "trailing_median_oos_mar": trailing_med,
        "latest_oos_mar": latest,
        "context_features_topk": _topk_features(topk_from_csv, k=k),
        "negative_patterns": _negative_patterns(),
        "guidance": "Bias generation toward configurations exploiting top-K stable features; avoid negative patterns."
    }
    out_json = os.path.join(out_dir, "llm_prompt.json")
    with open(out_json,"w",encoding="utf-8") as f: json.dump(prompt, f, ensure_ascii=False, indent=2)

    mlflow.set_experiment("reinforcement_reward")
    with mlflow.start_run(run_name=f"reward_{strategy}_{ts}") as run:
        mlflow.set_tags({"phase":"11","module":"reward","strategy":strategy})
        mlflow.log_metric("reward_scalar", float(prompt["reward_scalar"]))
        mlflow.log_artifact(out_json, artifact_path="reward")

    # try to tag latest source run as logged
    try:
        r = _latest_run_for_strategy(client, primary, strategy)
        if r:
            client.set_tag(r.info.run_id, "reward_logged", "yes")
    except Exception:
        pass

    return {"out_dir": out_dir, "json": out_json, "reward_scalar": prompt["reward_scalar"]}
