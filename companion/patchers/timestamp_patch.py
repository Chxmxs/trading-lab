"""
timestamp_patch.py
------------------
Fixes common timestamp problems in dataframes:
 - Renames date/time columns to 'timestamp'
 - Converts to UTC DatetimeIndex
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def patch(df):
    """
    Input: pandas.DataFrame
    Output: patched DataFrame (or same DataFrame if already fine)
    """
    if df is None or not isinstance(df, pd.DataFrame):
        logger.warning("timestamp_patch: input is not a DataFrame")
        return df

    # 1. Try to find a column that looks like a timestamp
    for col in df.columns:
        if col.lower() in ["time", "date", "datetime", "timestamp"]:
            df = df.rename(columns={col: "timestamp"})
            break

    # 2. Ensure timestamp column exists
    if "timestamp" not in df.columns:
        logger.error("timestamp_patch: No timestamp column found.")
        return df

    # 3. Convert to datetime and set as index (auto-detect epoch seconds)
    try:
        ts = df["timestamp"]
        # If it's integer-like and looks like Unix seconds, set unit='s'
        unit = None
        if pd.api.types.is_integer_dtype(ts) or pd.api.types.is_float_dtype(ts):
            # crude check: 10-digit values in typical Unix seconds range
            # avoid year 1970 from ns interpretation
            if ts.dropna().astype(str).str.len().median() in (10, 13):
                unit = "s" if ts.astype(str).str.len().median() == 10 else "ms"
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", unit=unit)
        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        logger.info("timestamp_patch: UTC index set (unit=%s).", unit or "auto")
    except Exception as e:
        logger.exception("timestamp_patch failed: %s", e)


    return df
