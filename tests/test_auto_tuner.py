# -*- coding: utf-8 -*-
from pathlib import Path
import json
from companion.ml.auto_tuner import suggest_from_trials

def test_suggest_from_trials(tmp_path, monkeypatch):
    # create a tiny trials csv in tmp
    p = tmp_path / "trials.csv"
    p.write_text("oos_mar,a,b\n1.1,5,20\n0.9,3,15\n1.3,7,25\n1.0,6,10\n1.4,8,30\n1.2,9,22\n", encoding="utf-8")
    # disable real mlflow writing by pointing to tmp folder (file://)
    import os
    os.environ["MLFLOW_TRACKING_URI"] = "file:///" + str(tmp_path).replace("\\","/") + "/mlruns"
    out = suggest_from_trials(str(p))
    # artifacts exist
    assert "artifacts_dir" in out
    adir = Path(out["artifacts_dir"])
    assert (adir / "feature_importances.csv").exists()
    assert (adir / "suggestions.json").exists()
    # load suggestions.json schema check
    s = json.loads((adir / "suggestions.json").read_text("utf-8"))
    assert "top_k_rows" in s and "bounds" in s
