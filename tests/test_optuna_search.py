import importlib
import types

def test_run_optuna_search_monkeypatch(monkeypatch, tmp_path):
    # Import modules
    opt_mod = importlib.import_module("companion.explorer.optuna_search")
    disc_mod = importlib.import_module("companion.explorer.discovery")
    eval_mod = importlib.import_module("companion.explorer.evaluation")

    # Mock evaluator: returns metric based on 'lookback' to give Optuna a slope
    def fake_eval(candidate=None, config=None, data_context=None, *a, **k):
        lb = candidate.params.get("lookback", 50)
        thr = float(candidate.params.get("threshold", 1.2))
        # Simple target: higher lookback is better up to 150, penalize big thresholds slightly
        score = (min(lb, 150) / 150.0) - (thr - 1.0) * 0.1
        return {"sharpe": score, "mar": score - 0.05, "cagr": score - 0.1}

    monkeypatch.setattr(eval_mod, "evaluate_strategy_cv", fake_eval)

    # Minimal config (points to an existing strategy file/class in your repo)
    cfg = {
        "optuna_enabled": True,
        "optuna": {
            "study_name": "test_explorer",
            "direction": "maximize",
            "n_trials": 5,
            "sampler": "tpe",
            "pruner": "median",
            "metric": "sharpe",
            "candidate_template": {
                "base_module_file": "companion/strategies/breakout_strategy.py",
                "base_class_name": "BreakoutStrategy"
            },
            "search_space": {
                "lookback": {"type": "int", "low": 10, "high": 200},
                "threshold": {"type": "float", "low": 1.02, "high": 1.50, "step": 0.01}
            }
        }
    }

    best_cand, best_metrics = opt_mod.run_optuna_search(config=cfg, data_context={"data_dir": str(tmp_path)})
    assert best_cand is not None
    assert isinstance(best_metrics, dict)
    assert "params" in best_metrics
    assert "sharpe" in best_metrics  # from fake_eval
