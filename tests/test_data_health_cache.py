# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path

from companion.data_checks.health_runner import preflight_data_health

def test_preflight_caches(tmp_path, monkeypatch):
    # minimal fake sources
    f1 = tmp_path / "ohlcv.csv"
    f1.write_text("ts,open,high,low,close,volume\n", encoding="utf-8")
    f2 = tmp_path / "metric.csv"
    f2.write_text("ts,value\n", encoding="utf-8")

    # monkeypatch summarize_health to deterministic pass
    import companion.data_checks.health as health_mod
    def fake_sum(**kws):
        return {"status": "pass", "warnings": [], "errors": []}
    monkeypatch.setattr(health_mod, "summarize_health", fake_sum)

    out1 = preflight_data_health(
        symbol="BTCUSD",
        timeframe="15m",
        window={"start": "2020-01-01", "end": "2020-02-01"},
        sources=[str(f1), str(f2)],
        cache_root=str(tmp_path / "cache"),
    )
    assert out1["status"] == "pass"
    first_path = Path(out1["summary_path"])
    assert first_path.exists()

    # call again -> should reuse cache
    out2 = preflight_data_health(
        symbol="BTCUSD",
        timeframe="15m",
        window={"start": "2020-01-01", "end": "2020-02-01"},
        sources=[str(f1), str(f2)],
        cache_root=str(tmp_path / "cache"),
    )
    assert out2["summary_path"] == str(first_path)