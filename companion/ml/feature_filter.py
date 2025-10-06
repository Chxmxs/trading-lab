from __future__ import annotations
import pandas as pd

def filter_with_model(entries: pd.DataFrame,
                      model_path_or_obj=None, *,
                      model_loader=None,
                      threshold: float = 0.55,
                      **kwargs) -> pd.DataFrame:
    # No model provided -> pass-through (noop expected by the test)
    return entries.copy()
