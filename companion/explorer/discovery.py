from __future__ import annotations
from typing import Dict, Any, Iterable, Mapping, Optional
import os, json

def build_prompt_context(strategies: Iterable[str] = (), symbols: Iterable[str] = (), timeframes: Iterable[str] = (), extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "strategies": list(strategies) if strategies is not None else [],
        "symbols": list(symbols) if symbols is not None else [],
        "timeframes": list(timeframes) if timeframes is not None else [],
    }
    # Discover from data/cleaned/ohlcv_<SYMBOL>@<TF>.csv
    available_symbols = []
    timeframes_by_symbol: Dict[str, list] = {}
    try:
        root = os.path.join(os.getcwd(), "data", "cleaned")
        if os.path.isdir(root):
            for name in os.listdir(root):
                n = name.lower()
                if not (n.startswith("ohlcv_") and n.endswith(".csv")):
                    continue
                # ohlcv_BTCUSD@15m.csv -> symbol=BTCUSD, tf=15m
                core = name[6:-4]
                parts = core.split("@")
                symbol = parts[0] if parts else ""
                tf = parts[1] if len(parts) > 1 else ""
                if symbol:
                    if symbol not in available_symbols:
                        available_symbols.append(symbol)
                    if symbol not in timeframes_by_symbol:
                        timeframes_by_symbol[symbol] = []
                    if tf and tf not in timeframes_by_symbol[symbol]:
                        timeframes_by_symbol[symbol].append(tf)
        # sort outputs for determinism
        available_symbols.sort()
        for s in timeframes_by_symbol.keys():
            timeframes_by_symbol[s] = sorted(timeframes_by_symbol[s])
    except Exception:
        pass
    ctx["available_symbols"] = available_symbols
    ctx["timeframes_by_symbol"] = timeframes_by_symbol
    if extra:
        ctx.update(dict(extra))
    return ctx

def enrich_prompt_with_context(prompt: str, context: Mapping[str, Any]) -> str:
    lines = [prompt.rstrip(), "", "[CONTEXT]"]
    for k in sorted(context.keys()):
        lines.append(f"{k}: {context[k]}")
    return "\n".join(lines)

def _llm_chat(prompt: str, model: str, temperature: float = 0.2, max_tokens: int = 400) -> str:
    return json.dumps({"name":"Example","module_file":"companion/strategies/example.py","class_name":"Example","params":{}})

def propose_new_strategy_via_llm(prompt: str, *, model: str = "gpt-4o-mini", temperature: float = 0.2, max_tokens: int = 400, logs_path: Optional[str] = None, schema: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    raw = _llm_chat(prompt, model, temperature=temperature, max_tokens=max_tokens)
    if logs_path:
        try:
            with open(logs_path, "a", encoding="utf-8") as f:
                f.write(raw + "\n")
        except Exception:
            pass
    obj = json.loads(raw)
    for k in ("name","module_file","class_name","params"):
        if k not in obj:
            raise ValueError(f"missing key: {k}")
    return obj
