# encoding: utf-8
import os, json, pandas as pd, mlflow
from companion.explorer.auto_data_select import emit_selection

def test_auto_data_selection(tmp_path):
    ranks = tmp_path / "importance_rank.csv"
    pd.DataFrame([
        {"symbol":"BTCUSD","timeframe":"15m","feature":"asopr","mean_importance":0.7,"runs":3,"rank":1},
        {"symbol":"BTCUSD","timeframe":"15m","feature":"sopr","mean_importance":0.3,"runs":3,"rank":2},
        {"symbol":"BTCUSD","timeframe":"15m","feature":"stale_metric","mean_importance":0.9,"runs":1,"rank":3},
    ]).to_csv(ranks, index=False)
    health = tmp_path / "health.json"
    # stale_metric is stale/bad; asopr ok; sopr warn but acceptable
    (tmp_path / "artifacts" / "auto_data").mkdir(parents=True, exist_ok=True)
    h = {
      "asopr":{"status":"ok","updated_at":"2025-09-01T00:00:00Z"},
      "sopr":{"status":"warn","updated_at":"2025-09-10T00:00:00Z"},
      "stale_metric":{"status":"bad","updated_at":"2020-01-01T00:00:00Z"}
    }
    health.write_text(json.dumps(h), encoding="utf-8")
    uri = "file:///" + str(tmp_path.joinpath("mlruns")).replace("\\","/")
    res = emit_selection("SOPRRegimeBand", str(ranks), str(health), K=2, staleness_days=3650, out_root=str(tmp_path/"artifacts"/"auto_data"), mlflow_tracking_uri=uri)
    assert os.path.exists(res["json"])
    data = json.load(open(res["json"],"r",encoding="utf-8"))
    assert data and data[0]["metrics"] == ["asopr","sopr"]
