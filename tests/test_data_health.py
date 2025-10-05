# -*- coding: utf-8 -*-
import pandas as pd
from companion.data_checks.health import check_missing_bars, check_duplicates, check_monotonic_utc, summarize_health

def test_health_pass_basic():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
        "open": [1,2,3,4,5]
    })
    s1,_ = check_missing_bars(df)
    s2,_ = check_duplicates(df)
    s3,_ = check_monotonic_utc(df)
    s_all,_ = summarize_health(df)
    assert s1 == "pass"
    assert s2 == "pass"
    assert s3 in ("pass","warn")  # monotonic should be pass
    assert s_all in ("pass","warn")

def test_health_warn_duplicates():
    df = pd.DataFrame({"timestamp":[1,1,2], "x":[1,1,2]})
    s,_ = check_duplicates(df)
    assert s in ("warn","fail")

def test_health_fail_missing_ts():
    df = pd.DataFrame({"x":[1,2,3]})
    s,_ = check_missing_bars(df)
    assert s == "fail"
