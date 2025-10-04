from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any
from pathlib import Path
import importlib.util
import sys


@dataclass
class StrategyCandidate:
    # Core fields used by the explorer
    name: str
    func: Callable
    params: Dict[str, Any]
    source: str = "manual"

    # Optional metadata (useful for LLM / loaders)
    module_file: str | None = None
    class_name: str | None = None


def load_strategy_callable_from_module(module_file: str, class_name: str) -> Callable:
    """
    Dynamically import a Python module by file path and adapt a Strategy class
    into a callable with signature: func(df, run_config, **params) -> dict
    The Strategy class must implement __init__(**params) and run(df, run_config).
    """
    path = Path(module_file)
    if not path.is_file():
        raise FileNotFoundError(f"Module file not found: {module_file}")

    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from: {module_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)  # type: ignore

    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {module_file}")

    cls = getattr(module, class_name)

    def strategy_callable(df, run_config, **params):
        instance = cls(**params)
        return instance.run(df, run_config)

    return strategy_callable
