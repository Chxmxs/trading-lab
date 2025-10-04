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

    # 3. Convert to datetime and set as index
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        logger.info("timestamp_patch: Successfully converted and set UTC index.")
    except Exception as e:
        logger.exception("timestamp_patch failed: %s", e)

    return df
