import importlib
import json

def test_llm_mock_strategy(monkeypatch, tmp_path):
    discovery = importlib.import_module("companion.explorer.discovery")

    # Fake LLM response
    fake_json = json.dumps({
        "name": "LLMTest",
        "module_file": "companion/strategies/breakout_strategy.py",
        "class_name": "BreakoutStrategy",
        "params": {"lookback": 30, "threshold": 1.15}
    })

    def fake_llm_chat(prompt, model, temperature=0.2, max_tokens=400):
        return fake_json

    monkeypatch.setattr(discovery, "_llm_chat", fake_llm_chat)

    cfg = {
        "llm_enabled": True,
        "llm": {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 400,
            "logs_path": str(tmp_path / "llm_logs.json")
        }
    }

    cand = discovery.propose_new_strategy_via_llm(cfg)
    assert cand is not None
    assert cand.name == "LLMTest"
    assert "lookback" in cand.params
