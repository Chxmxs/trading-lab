"""
scoring.py
----------
Custom composite scoring for strategy ranking.
Handles both legacy and new naming styles for drawdown & trade weights.
"""

import logging
logger = logging.getLogger(__name__)

def custom_score(metrics, weights):
    """
    Combine metrics with user-defined weights.
    Supports flexible naming:
      weights: may include either 'drawdown' or 'max_dd_penalty'
      metrics: may include either 'max_drawdown' or 'max_dd'
    """
    # Defensive lookups with fallbacks
    dd_weight = (
        weights.get("drawdown")
        or -weights.get("max_dd_penalty", 0.0)
    )
    trades_weight = weights.get("trades", 0.0)

    sharpe_w = weights.get("sharpe", 0.0)
    mar_w = weights.get("mar", 0.0)
    cagr_w = weights.get("cagr", 0.0)

    # Extract metric names flexibly
    sharpe = metrics.get("sharpe", 0.0)
    mar = metrics.get("mar", 0.0)
    cagr = metrics.get("cagr", 0.0)
    max_dd = (
        metrics.get("max_drawdown")
        or metrics.get("max_dd")
        or 0.0
    )
    num_trades = metrics.get("num_trades", 0.0)

    score = (
        sharpe_w * sharpe
        + mar_w * mar
        + cagr_w * cagr
        + dd_weight * abs(max_dd)
        + trades_weight * num_trades
    )

    logger.debug(
        "Score calc: sharpe=%.3f mar=%.3f cagr=%.3f dd=%.3f trades=%.1f -> %.3f",
        sharpe, mar, cagr, max_dd, num_trades, score
    )

    return score
