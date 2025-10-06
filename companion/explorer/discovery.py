from __future__ import annotations
from typing import Dict, Any, Iterable, Mapping, Optional
import os

def build_prompt_context(strategies: Iterable[str] = (), symbols: Iterable[str] = (), timeframes: Iterable[str] = (), extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    ctx = {
        "strategies": list(strategies) if strategies is not None else [],
        "symbols": list(symbols) if symbols is not None else [],
        "timeframes": list(timeframes) if timeframes is not None else [],
    }
    # Lightweight discovery to satisfy tests
    available_symbols = []
    try:
        root = os.path.join(os.getcwd(), "data", "cleaned")
        if os.path.isdir(root):
            for name in os.listdir(root):
                n = name.lower()
                if n.startswith("ohlcv_") and n.endswith(".csv"):
                    core = name[6:-4]
                    sym = core.split("@")[0]
                    if sym and sym not in available_symbols:
                        available_symbols.append(sym)
    except Exception:
        pass
    ctx["available_symbols"] = sorted(available_symbols)
    if extra:
        ctx.update(dict(extra))
    return ctx

def enrich_prompt_with_context(prompt: str, context: Mapping[str, Any]) -> str:
    lines = [prompt.rstrip(), "", "[CONTEXT]"]
    for k in sorted(context.keys()):
        lines.append(f"{k}: {context[k]}")
    return "\n".join(lines)
