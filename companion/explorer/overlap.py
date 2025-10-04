from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict

def compute_equity_correlation(equity_a: pd.Series, equity_b: pd.Series) -> float:
    """
    Compute the Pearson correlation between two equity curves using returns.
    """
    df = pd.concat([equity_a.pct_change(), equity_b.pct_change()], axis=1, join="inner").dropna()
    if len(df) < 2:
        return 0.0
    corr = df.corr().iloc[0, 1]
    return float(corr)

def build_correlation_matrix(equity_series: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
    """
    Build a symmetric correlation matrix for multiple equity series.
    """
    names = list(equity_series.keys())
    matrix: Dict[str, Dict[str, float]] = {name: {} for name in names}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j < i:
                # Use symmetry
                matrix[a][b] = matrix[b][a]
            else:
                if a == b:
                    matrix[a][b] = 1.0
                else:
                    matrix[a][b] = compute_equity_correlation(equity_series[a], equity_series[b])
    return matrix
