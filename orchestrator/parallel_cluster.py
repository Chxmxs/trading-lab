# encoding: utf-8
import os, json, time, uuid, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import mlflow

class LocalCluster:
    def __init__(self, workdir="artifacts/cluster_runs"):
        self.workdir = workdir
        os.makedirs(self.workdir, exist_ok=True)
        self._pool = ProcessPoolExecutor(max_workers=2)
        self._handles = {}

    @staticmethod
    def _worker(job):
        out_dir = job.get("out_dir")
        os.makedirs(out_dir, exist_ok=True)
        # simulate artifact creation
        with open(os.path.join(out_dir,"artifact.txt"),"w",encoding="utf-8") as f:
            f.write("ok")
        time.sleep(0.05)
        return {"status":"completed","artifacts":[os.path.join(out_dir,"artifact.txt")]}

    def submit(self, job):
        handle = str(uuid.uuid4())
        out_dir = os.path.join(self.workdir, handle)
        job = dict(job); job["out_dir"] = out_dir
        fut = self._pool.submit(LocalCluster._worker, job)
        self._handles[handle] = {"future": fut, "out_dir": out_dir}
        return handle

    def poll(self, handle):
        h = self._handles[handle]
        if h["future"].done():
            return "completed"
        return "running"

    def fetch_artifacts(self, handle, dest):
        h = self._handles[handle]
        os.makedirs(dest, exist_ok=True)
        for f in os.listdir(h["out_dir"]):
            shutil.copy2(os.path.join(h["out_dir"], f), dest)
        return [os.path.join(dest, f) for f in os.listdir(dest)]

def submit_poll_fetch(jobs, out_root="artifacts/cluster_collated", mlflow_tracking_uri=None):
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("parallel_cluster")
    cluster = LocalCluster()
    handles = [cluster.submit(j) for j in jobs]
    done=set()
    while len(done)<len(handles):
        for h in handles:
            if h in done: continue
            if cluster.poll(h)=="completed":
                done.add(h)
        time.sleep(0.01)
    os.makedirs(out_root, exist_ok=True)
    collated=[]
    for h in handles:
        collated.extend(cluster.fetch_artifacts(h, out_root))
    with mlflow.start_run(run_name="local_fanout"):
        mlflow.set_tags({"phase":"11","module":"parallel_cluster"})
        for p in collated: mlflow.log_artifact(p, artifact_path="collated")
        mlflow.log_metric("jobs", len(jobs))
    return collated
