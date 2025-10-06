# encoding: utf-8
import os, json, random, math
from datetime import datetime, timezone
import mlflow

SEED=777
random.seed(SEED)

def _utc_ts(): return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def _load_json(path, default):
    if not path or not os.path.exists(path): return default
    return json.load(open(path,"r",encoding="utf-8"))

def mutate_params(base, schema):
    out = dict(base)
    for k, spec in schema.items():
        if spec.get("type")=="float":
            lo, hi = float(spec["low"]), float(spec["high"])
            span = (hi - lo)
            jitter = random.uniform(-0.1, 0.1) * span
            val = float(base.get(k, (lo+hi)/2.0)) + jitter
            out[k] = max(lo, min(hi, val))
        elif spec.get("type")=="int":
            lo, hi = int(spec["low"]), int(spec["high"])
            step = spec.get("step",1)
            val = int(base.get(k, (lo+hi)//2)) + random.choice([-step,0,step])
            out[k] = max(lo, min(hi, val))
        elif spec.get("type")=="bool":
            out[k] = bool(random.choice([True, False]))
        else:
            out[k] = base.get(k)
    return out

def evolve(master_list_path, suggestions_path, schema, N=10, out_root="artifacts/genetic", queue_path="artifacts/_queue/jobs.jsonl", mlflow_tracking_uri=None):
    master = _load_json(master_list_path, {"strategies":[]})
    sugg   = _load_json(suggestions_path, {"suggestions":[]})
    # pick seed parents from master top winners + suggestions
    parents = []
    for s in master.get("strategies", [])[:10]:
        if "params" in s: parents.append({"strategy": s.get("strategy"), "params": s["params"]})
    for s in sugg.get("suggestions", [])[:10]:
        if "params" in s and "strategy" in s: parents.append({"strategy": s["strategy"], "params": s["params"]})
    if not parents:
        # fallback dummy parent
        parents = [{"strategy":"SOPRRegimeBand","params":{k:(v.get("low") if v.get("type")!="bool" else False) for k,v in schema.items()}}]

    ts = _utc_ts()
    out_dir = os.path.join(out_root, ts)
    os.makedirs(out_dir, exist_ok=True)
    cand_path = os.path.join(out_dir, "candidates.jsonl")
    with open(cand_path,"w",encoding="utf-8") as f:
        for i in range(N):
            p = random.choice(parents)
            child_params = mutate_params(p["params"], schema)
            rec = {"strategy": p["strategy"], "params": child_params, "source":"ga", "gen": 1}
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    os.makedirs(os.path.dirname(queue_path), exist_ok=True)
    with open(queue_path,"a",encoding="utf-8") as q:
        with open(cand_path,"r",encoding="utf-8") as f:
            for line in f:
                job = {"job_type":"ga_candidate","status":"pending","payload":json.loads(line)}
                q.write(json.dumps(job, ensure_ascii=False)+"\n")

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("strategy_evolution")
    with mlflow.start_run(run_name=f"ga_{ts}"):
        mlflow.set_tags({"phase":"11","module":"genetic"})
        mlflow.log_artifact(cand_path, artifact_path="genetic")
        mlflow.log_metric("candidates", N)
    return {"out_dir": out_dir, "candidates": cand_path, "enqueued": True}
