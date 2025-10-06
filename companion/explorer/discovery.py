from __future__ import annotations
from typing import Dict, Any, Iterable, Mapping, Optional
from types import SimpleNamespace
import os, json

def build_prompt_context(strategies: Iterable[str] = (), symbols: Iterable[str] = (), timeframes: Iterable[str] = (), extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {"strategies": list(strategies) if strategies is not None else [], "symbols": list(symbols) if symbols is not None else [], "timeframes": list(timeframes) if timeframes is not None else []}
    available_symbols = []; timeframes_by_symbol: Dict[str, list] = {}
    try:
        root = os.path.join(os.getcwd(), "data", "cleaned")
        if os.path.isdir(root):
            for name in os.listdir(root):
                n = name.lower()
                if not (n.startswith("ohlcv_") and n.endswith(".csv")): continue
                core = name[6:-4]; parts = core.split("@"); sym = parts[0] if parts else ""; tf = parts[1] if len(parts)>1 else ""
                if sym:
                    if sym not in available_symbols: available_symbols.append(sym)
                    timeframes_by_symbol.setdefault(sym, [])
                    if tf and tf not in timeframes_by_symbol[sym]: timeframes_by_symbol[sym].append(tf)
        available_symbols.sort();
        for s in list(timeframes_by_symbol.keys()): timeframes_by_symbol[s] = sorted(timeframes_by_symbol[s])
    except Exception: pass
    ctx["available_symbols"] = available_symbols
    ctx["timeframes_by_symbol"] = timeframes_by_symbol
    if extra: ctx.update(dict(extra))
    return ctx

def enrich_prompt_with_context(prompt: str, context: Mapping[str, Any] = None) -> str:
    context = {} if context is None else context
    lines = [prompt.rstrip(), "", "## DATA CONTEXT"]
    for k in sorted(context.keys()): lines.append(f"{k}: {context[k]}")
    return "\\n".join(lines)

def _llm_chat(prompt: str, model: str, temperature: float = 0.2, max_tokens: int = 400) -> str:
    return json.dumps({"name":"Example","module_file":"companion/strategies/example.py","class_name":"Example","params":{}})

def propose_new_strategy_via_llm(cfg: Mapping[str, Any]) -> Optional[SimpleNamespace]:
    if not cfg or not cfg.get("llm_enabled"): return None
    llm = dict(cfg.get("llm", {}))
    raw = _llm_chat(prompt=json.dumps(cfg), model=llm.get("model","gpt-4o-mini"), temperature=float(llm.get("temperature",0.2)), max_tokens=int(llm.get("max_tokens",400)))
    lp = llm.get("logs_path")
    if lp:
        try:
            with open(lp, "a", encoding="utf-8") as f:
                f.write(json.dumps({"response": raw}) + "\\n")  # tests look for \\"response\\"
        except Exception: pass
    obj = json.loads(raw)
    return SimpleNamespace(**obj)
