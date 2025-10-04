import json
import pandas as pd
from companion.explorer.overlap import build_correlation_matrix

with open("equities.json", "r", encoding="utf-8") as f:
    eq_data = json.load(f)

equity_series = {name: pd.Series(values) for name, values in eq_data.items()}
matrix = build_correlation_matrix(equity_series)
for strategy_a, correlations in matrix.items():
    print(strategy_a, correlations)
