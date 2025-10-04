from __future__ import annotations

import importlib.util
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class StrategyCandidate:
    name: str
    func: Callable
    params: Dict[str, Any] = field(default_factory=dict)
    source: str = "manual"

def generate_parameter_sweep(
    base_strategy: Callable,
    search_space: Dict[str, Iterable[Any]],
    max_candidates: int,
    name_prefix: str = "sweep",
) -> List[StrategyCandidate]:
    keys = list(search_space.keys())
    values_list = [list(search_space[k]) for k in keys]
    all_combinations: List[Tuple[Any, ...]] = []
    total_combinations = 1
    for vals in values_list:
        total_combinations *= len(vals)
    if total_combinations <= max_candidates:
        import itertools
        for combo in itertools.product(*values_list):
            all_combinations.append(combo)
    else:
        for _ in range(max_candidates):
            combo = tuple(random.choice(values_list[i]) for i in range(len(keys)))
            all_combinations.append(combo)
    candidates: List[StrategyCandidate] = []
    for idx, combo in enumerate(all_combinations):
        params = dict(zip(keys, combo))
        def strategy_wrapper(df, run_config, _params=params, _fn=base_strategy):
            return _fn(df, run_config, **_params)
        name = f"{name_prefix}_{idx}"
        candidates.append(StrategyCandidate(name=name, func=strategy_wrapper, params=params, source="sweep"))
    return candidates

def mutate_strategy(
    strategy: StrategyCandidate,
    perturbation: Dict[str, Tuple[float, float]],
    max_mutations: int = 1,
) -> StrategyCandidate:
    base_params = strategy.params.copy()
    mutated_params = base_params.copy()
    for _ in range(max_mutations):
        for param, (low, high) in perturbation.items():
            if param in mutated_params and isinstance(mutated_params[param], (int, float)):
                factor = random.uniform(low, high)
                mutated_params[param] = type(mutated_params[param])(mutated_params[param] * factor)
    def mutated_func(df, run_config, _params=mutated_params, _fn=strategy.func):
        return _fn(df, run_config, **_params)
    mutated_name = f"{strategy.name}_mut"
    return StrategyCandidate(name=mutated_name, func=mutated_func, params=mutated_params, source="mutation")

def propose_new_strategy_via_llm(llm_client, prompt_template, tools=None, temperature=0.5):
    """
    Placeholder LLM integration. Replace with real implementation if needed.
    """
    raise NotImplementedError("LLM integration is not yet implemented.")

def register_strategy_from_script(script_path: str) -> Optional[StrategyCandidate]:
    path = Path(script_path)
    if not path.is_file():
        logger.error("Strategy script %s does not exist.", script_path)
        return None
    try:
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)  # type: ignore
        strategy_func = getattr(module, "strategy", None)
        if not callable(strategy_func):
            logger.error("No callable 'strategy' found in %s", script_path)
            return None
        candidate_name = path.stem
        return StrategyCandidate(name=candidate_name, func=strategy_func, params={}, source="manual")
    except Exception as exc:
        logger.exception("Failed to register strategy from %s: %s", script_path, exc)
        return None
