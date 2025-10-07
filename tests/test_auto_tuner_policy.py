# -*- coding: utf-8 -*-
import json
from pathlib import Path
from companion.ml.auto_tuner import Policy

def test_policy_load_defaults(tmp_path: Path):
    p = Policy.load(tmp_path / "missing.json")
    assert p.min_trials > 0
    cfg = tmp_path / "policy.json"
    cfg.write_text(json.dumps({"top_frac": 0.5, "kfolds": 3}), encoding="utf-8")
    q = Policy.load(cfg)
    assert q.top_frac == 0.5
    assert q.kfolds == 3
