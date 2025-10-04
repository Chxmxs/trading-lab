import json
import importlib
from pathlib import Path

# Import the module under test
mod = importlib.import_module("companion.explorer.discovery")

def test_propose_new_strategy_via_llm_monkeypatch(tmp_path, monkeypatch):
    # Prepare a fake config enabling LLM + logging to temp file
    logs_path = tmp_path / "llm_logs.json"
    cfg = {
        "llm_enabled": True,
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 400,
            "logs_path": str(logs_path),
            "strategy_json_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "module_file": {"type": "string"},
                    "class_name": {"type": "string"},
                    "params": {"type": "object"}
                },
                "required": ["name", "module_file", "class_name", "params"]
            }
        }
    }

    # Mock the LLM raw response with valid JSON
    fake_json = json.dumps({
        "name": "LLMBreakout",
        "module_file": "companion/strategies/breakout_strategy.py",
        "class_name": "BreakoutStrategy",
        "params": {"lookback": 50, "threshold": 1.2}
    })

    def fake_llm(prompt, model, temperature=0.2, max_tokens=400):
        return fake_json

    monkeypatch.setattr(mod, "_llm_chat", fake_llm)

    cand = mod.propose_new_strategy_via_llm(cfg)
    assert cand is not None
    assert cand.name == "LLMBreakout"
    assert cand.module_file.endswith("companion/strategies/breakout_strategy.py")
    assert cand.class_name == "BreakoutStrategy"
    assert isinstance(cand.params, dict)
    assert "lookback" in cand.params

    # Check that logs were written
    assert logs_path.exists()
    lines = logs_path.read_text(encoding="utf-8").strip().splitlines()
    assert any('"response"' in ln for ln in lines)
