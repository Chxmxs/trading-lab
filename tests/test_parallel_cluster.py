# encoding: utf-8
import os, mlflow
from orchestrator.parallel_cluster import submit_poll_fetch

def test_parallel_cluster_local(tmp_path):
    uri = "file:///" + str(tmp_path.joinpath("mlruns")).replace("\\","/")
    outs = submit_poll_fetch([{"name":"job1"},{"name":"job2"}], out_root=str(tmp_path/"artifacts"/"collated"), mlflow_tracking_uri=uri)
    assert len(outs)>=2
    for p in outs: assert os.path.exists(p)
