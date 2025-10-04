from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
from companion.explorer.scoring import custom_score

logger = logging.getLogger(__name__)

@dataclass
class StrategyResult:
    name: str
    metrics: Dict[str, float]
    source: str
    rank: Optional[int] = None
    score: Optional[float] = None

def compute_score(metrics: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """
    Compute a composite score based on supplied weights. If no weights are provided,
    defaults are used.
    """
    default_weights = {
        "sharpe": 1.5,
        "mar": 1.0,
        "cagr": 0.5,
        "drawdown": -2.0,
        "trades": 0.1,
    }
    w = weights if weights is not None else default_weights
    return custom_score(metrics, w)


def load_master_list(path: str) -> List[StrategyResult]:
    p = Path(path)
    if not p.is_file():
        return []
    with p.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    results = []
    for item in data:
        results.append(StrategyResult(**item))
    return results

def save_master_list(path: str, results: List[StrategyResult]) -> None:
    p = Path(path)
    serializable = [asdict(r) for r in results]
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8-sig") as f:
        json.dump(serializable, f, indent=2)

def update_master_list(
    master_path: str,
    new_results: List[StrategyResult],
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    correlation_threshold: float = 0.95,
) -> List[StrategyResult]:
    existing = {r.name: r for r in load_master_list(master_path)}
    for res in new_results:
        res.score = compute_score(res.metrics)
        if res.name in existing:
            if (existing[res.name].score or -float("inf")) < res.score:
                logger.info("Updating strategy %s with improved score.", res.name)
                existing[res.name] = res
        else:
            existing[res.name] = res
    # Basic correlation logic (optional)
    if correlation_matrix:
        to_remove: set[str] = set()
        names = list(existing.keys())
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                corr = correlation_matrix.get(a, {}).get(b)
                if corr is not None and corr >= correlation_threshold:
                    a_score = existing[a].score or compute_score(existing[a].metrics)
                    b_score = existing[b].score or compute_score(existing[b].metrics)
                    if a_score >= b_score:
                        to_remove.add(b)
                    else:
                        to_remove.add(a)
        for name in to_remove:
            logger.info("Removing overlapping strategy %s due to high correlation.", name)
            existing.pop(name, None)
    sorted_results = sorted(existing.values(), key=lambda r: r.score or 0.0, reverse=True)
    for rank, res in enumerate(sorted_results, start=1):
        res.rank = rank
    save_master_list(master_path, sorted_results)
    return sorted_results
