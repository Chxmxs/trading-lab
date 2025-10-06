from __future__ import annotations
from typing import Dict

def classify_error(text: str) -> str:
    if not text:
        return "unknown"
    t = text.lower()
    if "unicode" in t or "utf-8" in t or "utf-16" in t: return "encoding"
    if "json" in t and ("bom" in t or "utf" in t): return "data_bom"
    if "filenotfound" in t or "no such file" in t or "cannot find path" in t: return "missing_file"
    if "mlflow" in t and ("uri" in t or "tracking" in t): return "mlflow_uri"
    if "min trades" in t or "min_trades" in t: return "min_trades"
    if "connectionerror" in t or "timeout" in t or "temporarily unavailable" in t: return "transient"
    if "permission" in t or "access is denied" in t: return "permission"
    return "unknown"

def is_transient(label: str) -> bool:
    return label in {"transient"}
