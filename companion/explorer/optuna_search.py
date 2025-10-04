from __future__ import annotations

import importlib.util
import sys
from typing import Dict, Any, Tuple

# Optuna
import optuna

# Local imports: adjust paths if your package layout differs
from companion.explorer.common import StrategyCandidate, load_strategy_callable_from_module
from companion.explorer import evaluation  # must expose evaluate_strategy_cv(data, candidate, config) or similar


# ---------------------------
# Sampling helpers
# ---------------------------

def _sample_param(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    """
    Supports basic types: int, float, categorical.
    Spec example:
      {"type":"int","low":10,"high":200}
      {"type":"float","low":0.1,"high":2.0,"step":0.01}
      {"type":"categorical","choices":[10,20,50]}
    """
    ptype = str(spec.get("type", "float")).lower()

    if ptype == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        return trial.suggest_int(name, low, high)
    elif ptype == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        step = spec.get("step")
        if step is not None:
            step = float(step)
            # Discretized float
            n_steps = int(round((high - low) / step)) or 1
            idx = trial.suggest_int(f"{name}__idx", 0, n_steps)
            return low + idx * step
        else:
            return trial.suggest_float(name, low, high)
    elif ptype == "categorical":
        choices = spec["choices"]
        return trial.suggest_categorical(name, choices)
    else:
        raise ValueError(f"Unsupported param type: {ptype} for {name}")


def _build_candidate_from_config(search_space: Dict[str, Any],
                                 base_module_file: str,
                                 base_class_name: str,
                                 trial: optuna.Trial) -> StrategyCandidate:
    """
    Create a StrategyCandidate from base class and sampled params.
    """
    # Load base callable from class
    base_callable = load_strategy_callable_from_module(base_module_file, base_class_name)

    # Sample params
    params = {}
    for pname, spec in (search_space or {}).items():
        params[pname] = _sample_param(trial, pname, spec)

    # Wrap into project-expected callable signature
    def strategy_wrapper(df, run_config, _params=params, _fn=base_callable):
        return _fn(df, run_config, **_params)

    name = f"optuna_{trial.number}"
    return StrategyCandidate(name=name, func=strategy_wrapper, params=params, source="optuna",
                             module_file=base_module_file, class_name=base_class_name)


# ---------------------------
# Objective
# ---------------------------

def make_objective(config: Dict[str, Any], data_context: Dict[str, Any]):
    """
    Returns a function(trial) -> float that Optuna can optimize.
    - config: explorer config dict (must contain 'optuna' section)
    - data_context: whatever your evaluation.evaluate_strategy_cv expects to locate data.
      Typically { "data_dir": "...", ... } or you can pass None if evaluation loads internally.
    """
    opt_cfg = (config or {}).get("optuna") or {}
    metric_name = opt_cfg.get("metric", "sharpe")
    search_space = opt_cfg.get("search_space") or {}

    base_module_file = (opt_cfg.get("candidate_template") or {}).get("base_module_file")
    base_class_name = (opt_cfg.get("candidate_template") or {}).get("base_class_name")
    if not base_module_file or not base_class_name:
        raise ValueError("optuna.candidate_template.base_module_file and base_class_name are required.")

    def objective(trial: optuna.Trial) -> float:
        # 1) Sample a candidate
        cand = _build_candidate_from_config(
            search_space=search_space,
            base_module_file=base_module_file,
            base_class_name=base_class_name,
            trial=trial,
        )

        # 2) Evaluate via your existing CV evaluator
        # Your evaluation function should return a dict of metrics:
        #   {"sharpe": ..., "mar": ..., "cagr": ..., "dd": ..., "trade_count": ...}
        try:
            result = evaluation.evaluate_strategy_cv(candidate=cand, config=config, data_context=data_context)
        except TypeError:
            # Backwards-compat if your function signature differs
            result = evaluation.evaluate_strategy_cv(cand, config, data_context)

        # 3) Choose the metric to optimize
        if metric_name not in result:
            raise RuntimeError(f"Metric '{metric_name}' not found in evaluation results: {list(result.keys())}")
        score = float(result[metric_name])

        # Report intermediate values for pruners
        trial.report(score, step=0)
        return score

    return objective


# ---------------------------
# Entrypoint used by discovery.py
# ---------------------------

def run_optuna_search(config: Dict[str, Any], data_context: Dict[str, Any]) -> Tuple[StrategyCandidate, Dict[str, Any]]:
    """
    Executes an Optuna study and returns (best_candidate, best_metrics).
    Also returns the best candidate's params inside best_metrics['params'].
    """
    opt_cfg = (config or {}).get("optuna") or {}
    study_name = opt_cfg.get("study_name", "explorer_search")
    direction = opt_cfg.get("direction", "maximize").lower()
    n_trials = int(opt_cfg.get("n_trials", 30))
    timeout_sec = int(opt_cfg.get("timeout_sec", 0))

    # Sampler
    sampler_name = str(opt_cfg.get("sampler", "tpe")).lower()
    if sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler()
    else:
        sampler = optuna.samplers.TPESampler()  # default

    # Pruner
    pruner_name = str(opt_cfg.get("pruner", "median")).lower()
    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner()
    else:
        pruner = optuna.pruners.MedianPruner()

    # Create study
    study = optuna.create_study(study_name=study_name,
                                direction=direction,
                                sampler=sampler,
                                pruner=pruner)

    objective = make_objective(config, data_context)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec if timeout_sec > 0 else None)

    # Best trial → reconstruct candidate and re-evaluate to capture full metrics
    best = study.best_trial

    # Rebuild candidate with the best params
    # We need to rebuild exactly as we did during sampling:
    base_module_file = (opt_cfg.get("candidate_template") or {}).get("base_module_file")
    base_class_name = (opt_cfg.get("candidate_template") or {}).get("base_class_name")

    # Prepare a StrategyCandidate using best params
    base_callable = load_strategy_callable_from_module(base_module_file, base_class_name)

    def strategy_wrapper(df, run_config, _params=best.params, _fn=base_callable):
        return _fn(df, run_config, **_params)

    best_candidate = StrategyCandidate(
        name=f"optuna_best_{best.number}",
        func=strategy_wrapper,
        params=best.params,
        source="optuna",
        module_file=base_module_file,
        class_name=base_class_name,
    )

    # Evaluate best with your evaluator to get final metrics
    try:
        best_metrics = evaluation.evaluate_strategy_cv(candidate=best_candidate, config=config, data_context=data_context)
    except TypeError:
        best_metrics = evaluation.evaluate_strategy_cv(best_candidate, config, data_context)

    # Attach the best params for convenience
    best_metrics = dict(best_metrics or {})
    best_metrics["params"] = dict(best.params or {})
    return best_candidate, best_metrics
