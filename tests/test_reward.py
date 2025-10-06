# encoding: utf-8
import os, mlflow, json, pandas as pd, time, random
from companion.ai_loop.reward import compute_and_log_reward

def test_reward_pipeline(tmp_path):
    # set MLflow to temp file store
    uri = "file:///" + str(tmp_path.joinpath("mlruns")).replace("\\","/")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("strategy training runs")
    # seed some runs with oos_mar and strategy tag
    strat = "SOPRRegimeBand"
    vals = [0.8, 0.7, 0.75, 0.6, 0.65]
    for v in vals:
        with mlflow.start_run():
            mlflow.set_tag("strategy", strat)
            mlflow.log_metric("oos_mar", v)
            time.sleep(0.01)
    # fake top-k CSV
    fr = tmp_path / "importance_rank.csv"
    pd.DataFrame([{"symbol":"BTCUSD","timeframe":"15m","feature":"asopr","mean_importance":0.7,"runs":3,"rank":1},
                  {"symbol":"BTCUSD","timeframe":"15m","feature":"sopr","mean_importance":0.3,"runs":3,"rank":2}]).to_csv(fr, index=False)
    res = compute_and_log_reward(strat, ["strategy training runs"], str(fr), k=2, trailing_n=3, mlflow_tracking_uri=uri, out_root=str(tmp_path/"artifacts"/"reward"))
    assert os.path.exists(res["json"])
    j = json.load(open(res["json"],"r",encoding="utf-8"))
    assert "reward_scalar" in j and "context_features_topk" in j
    assert j["context_features_topk"][:2] == ["asopr","sopr"]
