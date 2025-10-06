# encoding: utf-8
import os, json, shutil, pandas as pd, mlflow, tempfile
from companion.ml.feature_importance_tracker import run_tracker

def _write_demo(tmpdir):
    # two mini CSVs with (optional) symbol/timeframe columns
    d1 = os.path.join(tmpdir, "artifacts", "_ci", "fi_demo1", "BTCUSD@15m")
    d2 = os.path.join(tmpdir, "artifacts", "_ci", "fi_demo2", "ETHUSD@1h")
    os.makedirs(d1, exist_ok=True); os.makedirs(d2, exist_ok=True)
    pd.DataFrame([
        {"feature":"asopr", "importance":0.7, "symbol":"BTCUSD", "timeframe":"15m"},
        {"feature":"sopr",  "importance":0.3, "symbol":"BTCUSD", "timeframe":"15m"},
    ]).to_csv(os.path.join(d1,"feature_importances.csv"), index=False)
    pd.DataFrame([
        {"feature":"mvrv", "importance":0.6, "symbol":"ETHUSD", "timeframe":"1h"},
        {"feature":"asopr","importance":0.4, "symbol":"ETHUSD", "timeframe":"1h"},
    ]).to_csv(os.path.join(d2,"feature_importances.csv"), index=False)
    return [os.path.join(tmpdir,"artifacts")]

def test_feature_importance_tracker(tmp_path, monkeypatch):
    roots = _write_demo(str(tmp_path))
    mlruns = os.path.join(str(tmp_path), "mlruns")
    uri = "file:///" + mlruns.replace("\\","/")
    res = run_tracker(roots, mlflow_tracking_uri=uri, output_root=os.path.join(str(tmp_path),"artifacts","feature_rank"))
    assert os.path.exists(res["csv"])
    assert os.path.exists(res["png"])
    # CSV schema
    df = pd.read_csv(res["csv"])
    for col in ["symbol","timeframe","feature","mean_importance","runs","rank"]:
        assert col in df.columns
    # MLflow created
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    exps = {e.name for e in client.list_experiments()}
    assert "feature_importance_tracker" in exps
