def custom_score(metrics, weights):
    """
    Combine metrics with user-defined weights.
    Example: weights = {"sharpe": 1.5, "mar": 1.0, "cagr": 0.5, "drawdown": -2.0, "trades": 0.1}
    """
    return (
        weights["sharpe"] * metrics["sharpe"]
        + weights["mar"] * metrics["mar"]
        + weights["cagr"] * metrics["cagr"]
        + weights["drawdown"] * abs(metrics["max_drawdown"])
        + weights["trades"] * metrics["num_trades"]
    )
