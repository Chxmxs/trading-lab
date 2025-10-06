# encoding: utf-8
import os, json, mlflow, tempfile
from companion.explorer.genetic import evolve

def test_genetic(tmp_path):
    master = tmp_path / "master_list.json"
    master.write_text(json.dumps({"strategies":[{"strategy":"SOPRRegimeBand","params":{"ma_regime":100,"tp_pct":0.05,"sl_pct":0.04}}]},indent=2), encoding="utf-8")
    schema = {
        "ma_regime":{"type":"int","low":50,"high":200,"step":5},
        "tp_pct":{"type":"float","low":0.01,"high":0.2},
        "sl_pct":{"type":"float","low":0.01,"high":0.2},
        "use_filter":{"type":"bool"}
    }
    uri = "file:///" + str(tmp_path.joinpath("mlruns")).replace("\\","/")
    out = evolve(str(master), None, schema, N=5, out_root=str(tmp_path/"artifacts"/"genetic"), queue_path=str(tmp_path/"artifacts"/"_queue"/"jobs.jsonl"), mlflow_tracking_uri=uri)
    assert os.path.exists(out["candidates"])
    # bounds check + queue appended
    qs = open(tmp_path/"artifacts"/"_queue"/"jobs.jsonl","r",encoding="utf-8").read().strip().splitlines()
    assert len(qs)>=5
    for line in open(out["candidates"],"r",encoding="utf-8"):
        rec = json.loads(line)
        assert 50 <= rec["params"]["ma_regime"] <= 200
        assert 0.01 <= rec["params"]["tp_pct"] <= 0.2
