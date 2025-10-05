# -*- coding: utf-8 -*-
"""
cache.py — hash (strategy, params, symbol, tf, data_window) -> cached result.
Writes under artifacts/_cache/{hash}.json
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
from typing import Any, Dict

ROOT = Path("artifacts/_cache")
ROOT.mkdir(parents=True, exist_ok=True)

def run_key(strategy: str, params: Dict[str,Any], symbol: str, timeframe: str, window: str) -> str:
    blob = json.dumps({"s":strategy,"p":params,"sym":symbol,"tf":timeframe,"win":window}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def has(key: str) -> bool:
    return (ROOT / f"{key}.json").exists()

def get(key: str) -> Dict[str,Any] | None:
    p = ROOT / f"{key}.json"
    if not p.exists(): return None
    return json.loads(p.read_text(encoding="utf-8"))

def put(key: str, payload: Dict[str,Any]) -> Path:
    p = ROOT / f"{key}.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p
