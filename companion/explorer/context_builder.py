from __future__ import annotations
from typing import Dict, Any, Iterable, Mapping, Optional
from .discovery import build_prompt_context

def build_data_map(strategies: Iterable[str] = (), symbols: Iterable[str] = (), timeframes: Iterable[str] = (), extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return build_prompt_context(strategies=strategies, symbols=symbols, timeframes=timeframes, extra=extra)
