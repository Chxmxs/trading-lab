import pandas as pd
from companion.error_handling import capture_errors, retry_with_backoff
import os

def test_capture_errors_creates_quarantine(tmp_path):
    called = {"flag": False}

    @capture_errors(run_name="unit_test_fail")
    def fail_func():
        called["flag"] = True
        raise ValueError("intentional test error")

    fail_func()
    assert called["flag"] is True
    qpath = "artifacts/_quarantine"
    found = any("unit_test_fail" in f for f in os.listdir(qpath))
    assert found, "Quarantine folder should be created"

def test_retry_with_backoff(monkeypatch):
    count = {"tries": 0}

    @retry_with_backoff(max_retries=2, base_delay=1)
    def flake():
        count["tries"] += 1
        if count["tries"] < 2:
            raise RuntimeError("temporary fail")
        return 42

    result = flake()
    assert result == 42
    assert count["tries"] == 2
