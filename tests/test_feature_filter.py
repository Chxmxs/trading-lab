# -*- coding: utf-8 -*-
import pandas as pd
from companion.ml.feature_adapter import save_features, load_features, list_features
from companion.ml.entry_filter import filter_with_model

def test_feature_adapter_roundtrip(tmp_path, monkeypatch):
    # temp feature store under tmp
    from pathlib import Path
    import companion.ml.feature_adapter as fa
    monkeypatch.setattr(fa, "ROOT", Path(tmp_path/"parquet"))
    df = pd.DataFrame({"timestamp":["2024-01-01T00:00:00Z"], "x":[1]})
    p = save_features(df, "BTC", "15m", "toy")
    assert p.exists()
    got = load_features("BTC", "15m", "toy")
    assert list(got.columns)==["timestamp","x"]
    feats = list_features("BTC", "15m")
    assert "toy" in feats

def test_entry_filter_noop():
    entries = pd.DataFrame({
        "timestamp": ["2024-01-01T00:00:00Z","2024-01-01T00:15:00Z"],
        "signal": [1,1]
    })
    filtered = filter_with_model(entries, None, model_loader=None, threshold=0.55)
    assert len(filtered) == 2
