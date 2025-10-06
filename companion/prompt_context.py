# companion/prompt_context.py

from __future__ import annotations
from typing import Dict, Iterable

# If build_prompt_context lives elsewhere, update this import accordingly:
try:
    from companion.discovery_context import build_prompt_context
except Exception:
    # Fallback import paths used in this repo during CI; adjust if needed
    from discovery_context import build_prompt_context  # type: ignore


def _fmt_list(xs: Iterable[str]) -> str:
    xs = list(dict.fromkeys(x.strip() for x in xs if x and str(x).strip()))
    return ", ".join(sorted(xs)) if xs else "(none)"

def enrich_prompt_with_context(base_prompt: str, context: Dict | None = None) -> str:
    """
    Append a human-readable DATA CONTEXT block to `base_prompt`.

    Notes for CI/tests:
    - `context` is OPTIONAL. If None, we build it via build_prompt_context().
    - Must include a "## DATA CONTEXT" header so tests can smoke-check the block.
    """
    if context is None:
        context = build_prompt_context()

    lines = [base_prompt.rstrip(), "", "## DATA CONTEXT", ""]

    # Strategies discovered
    strategies = context.get("all_strategies", [])
    lines.append(f"- strategies: {_fmt_list(strategies)}")

    # Symbols and timeframes
    tbs = context.get("timeframes_by_symbol", {}) or {}
    if isinstance(tbs, dict) and tbs:
        pairs = []
        for sym in sorted(tbs.keys()):
            tfs = tbs.get(sym) or []
            pairs.append(f"{sym}: {_fmt_list(tfs)}")
        lines.append("- symbols/timeframes: " + ("; ".join(pairs) if pairs else "(none)"))
    else:
        lines.append("- symbols/timeframes: (none)")

    # Optional extras if present in context (harmless if missing)
    ds = context.get("dataset_window")
    if ds:
        lines.append(f"- dataset_window: {ds}")

    mlflow_uri = context.get("mlflow_tracking_uri")
    if mlflow_uri:
        lines.append(f"- mlflow_tracking_uri: {mlflow_uri}")

    envs = context.get("conda_envs")
    if isinstance(envs, dict) and envs:
        kv = "; ".join(f"{k}={v}" for k, v in sorted(envs.items()))
        lines.append(f"- conda_envs: {kv}")

    lines.append("")  # trailing newline
    return "\n".join(lines)
