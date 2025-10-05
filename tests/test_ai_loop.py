import json
from companion.ai_loop import monitor, queue


def test_classify_error():
    assert monitor.classify_error("KeyError: missing column") == "data"
    assert monitor.classify_error("ImportError: no module") == "import"
    assert monitor.classify_error("ValueError: wrong shape") == "schema"
    assert monitor.classify_error("") == "performance"


def test_job_queue(tmp_path):
    qfile = tmp_path / "jobs.jsonl"
    jq = queue.JobQueue(str(qfile))

    jq.add_job({"run_id": "1", "priority": 5})
    jq.add_job({"run_id": "2", "priority": 1})

    jobs = jq.list_jobs()
    assert jobs[0]["run_id"] == "2"  # highest priority first

    job = jq.pop_job()
    assert job["run_id"] == "2"
    job = jq.pop_job()
    assert job["run_id"] == "1"
    assert jq.pop_job() is None
