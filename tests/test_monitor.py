# -*- coding: utf-8 -*-
import json
from pathlib import Path
from companion.ai_loop.monitor import scan_and_classify

def test_scan_and_classify_creates_outputs(tmp_path):
    # Minimal dummy MLflow objects
    class DummyRun:
        class info:
            run_id = "r1"
            status = "FAILED"
        class data:
            tags = {"error_message": "FileNotFoundError: data missing"}

    class DummyClient:
        def get_experiment_by_name(self, name):
            return type("Exp", (), {"experiment_id": "1"})()
        def search_runs(self, exp_ids, query):
            return [DummyRun()]
        def set_tag(self, run_id, key, val):
            # no-op for test
            pass

    out = scan_and_classify(experiment_name="default", outdir=tmp_path, client=DummyClient())
    assert out["count_classified"] == 1
    # decisions.jsonl exists
    found = list(tmp_path.rglob("decisions.jsonl"))
    assert found, "decisions.jsonl not written"
    # summary.json exists
    found_summary = list(tmp_path.rglob("summary.json"))
    assert found_summary, "summary.json not written"
