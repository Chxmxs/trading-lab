from __future__ import annotations
import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from companion.explorer.scoring import custom_score

logger = logging.getLogger(__name__)

# =============================================================
# Load config weights once (safe fallback if file missing)
# =============================================================
def _load_config_weights() -> Dict[str, float]:
    cfg_path = Path("configs/explorer.json")
    default_weights = {
        "sharpe": 1.5,
        "mar": 1.0,
        "cagr": 0.5,
        "drawdown": -2.0,
        "trades": 0.1,
    }
    if not cfg_path.is_file():
        logger.warning("Config file not found (%s). Using default weights.", cfg_path)
        return default_weights
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        weights = data.get("weights", {})
        merged = {**default_weights, **weights}  # merge user-defined into defaults
        logger.info("Loaded weights from config: %s", merged)
        return merged
    except Exception as e:
        logger.exception("Failed to load weights from config: %s", e)
        return default_weights


CONFIG_WEIGHTS = _load_config_weights()


# =============================================================
# Data class
# =============================================================
@dataclass
class StrategyResult:
    name: str
    metrics: Dict[str, float]
    source: str
    rank: Optional[int] = None
    score: Optional[float] = None


# =============================================================
# Scoring logic
# =============================================================
def compute_score(metrics: Dict[str, float], weights: Dict[str, float] | None = None) -> float:
    """
    Compute a composite score using weights from config (fallback to defaults).
    """
    w = weights if weights is not None else CONFIG_WEIGHTS
    try:
        return custom_score(metrics, w)
    except Exception as e:
        logger.exception("Score computation failed for metrics=%s: %s", metrics, e)
        return 0.0


# =============================================================
# Master list helpers
# =============================================================
def load_master_list(path: str) -> List[StrategyResult]:
    p = Path(path)
    if not p.is_file():
        return []
    with p.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return [StrategyResult(**item) for item in data]


def save_master_list(path: str, results: List[StrategyResult]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable = [asdict(r) for r in results]
    with p.open("w", encoding="utf-8-sig") as f:
        json.dump(serializable, f, indent=2)


def update_master_list(
    master_path: str,
    new_results: List[StrategyResult],
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    correlation_threshold: float = 0.95,
    weights: Optional[Dict[str, float]] = None,
) -> List[StrategyResult]:
    """
    Update or append strategies in the master list, compute new scores using
    config-defined weights, and remove highly correlated entries.
    """
    w = weights or CONFIG_WEIGHTS
    existing = {r.name: r for r in load_master_list(master_path)}

    # Merge new results
    for res in new_results:
        res.score = compute_score(res.metrics, w)
        if res.name in existing:
            prev_score = existing[res.name].score or -math.inf
            if res.score > prev_score:
                logger.info("Updating strategy %s with improved score (%.4f → %.4f)",
                            res.name, prev_score, res.score)
                existing[res.name] = res
        else:
            existing[res.name] = res

    # Optional correlation filtering
    if correlation_matrix:
        to_remove = set()
        names = list(existing.keys())
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                corr = correlation_matrix.get(a, {}).get(b)
                if corr is not None and corr >= correlation_threshold:
                    a_score = existing[a].score or compute_score(existing[a].metrics, w)
                    b_score = existing[b].score or compute_score(existing[b].metrics, w)
                    if a_score >= b_score:
                        to_remove.add(b)
                    else:
                        to_remove.add(a)
        for name in to_remove:
            logger.info("Removing overlapping strategy %s (corr ≥ %.2f)", name, correlation_threshold)
            existing.pop(name, None)

    # Sort and re-rank
    sorted_results = sorted(existing.values(), key=lambda r: r.score or 0.0, reverse=True)
    for rank, res in enumerate(sorted_results, start=1):
        res.rank = rank

    save_master_list(master_path, sorted_results)
    logger.info("Master list updated with %d strategies.", len(sorted_results))
    return sorted_results
