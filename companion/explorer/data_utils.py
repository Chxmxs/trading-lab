import pandas as pd
from pathlib import Path
from companion.patch_registry import apply_all_patches
import logging

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

REQUIRED_OHLC = {"open", "high", "low", "close"}  # volume optional

def _looks_like_ohlc(df: pd.DataFrame) -> bool:
    cols = {str(c).strip().lower() for c in df.columns}
    return REQUIRED_OHLC.issubset(cols)

def load_data_dirs(directories, date_range=None):
    """
    Load all CSV/Parquet files from given directories.
    - Auto-patch & enforce UTC index
    - Only keep DataFrames that contain OHLC columns
    - Skip/Log non-OHLC tables (metrics, audits, etc.)
    """
    frames = []
    for d in directories:
        path = Path(d)
        if not path.exists():
            logger.warning("Data directory does not exist: %s", path)
            continue

        for file_path in path.glob("**/*"):
            if file_path.suffix.lower() not in (".csv", ".parquet"):
                continue

            # 0) Quick header sniff: skip obvious non-price files by name
            name = file_path.name.lower()
            if any(bad in name for bad in ("audit", "report", "metrics", "readme", "notes", "schema")):
                logger.info("Skipping non-price file by name: %s", file_path)
                continue

            # 1) Read file
            try:
                if file_path.suffix.lower() == ".csv":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_parquet(file_path)
            except Exception as e:
                logger.exception("Failed reading %s: %s", file_path, e)
                continue

            # 2) Auto-patch (timestamp + ohlc) and normalize UTC
            try:
                df = apply_all_patches(df)
                if hasattr(df, "index") and isinstance(df.index, pd.DatetimeIndex):
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC")
                    else:
                        df.index = df.index.tz_convert("UTC")
            except Exception as e:
                logger.exception("Auto-patching or UTC normalization failed for %s: %s", file_path, e)
                continue

            # 3) Schema filter: keep only OHLC tables
            if not _looks_like_ohlc(df):
                logger.info("Skipping non-OHLC frame (cols=%s) from %s", list(df.columns)[:10], file_path)
                continue

            # 4) Optional date clip
            if date_range:
                start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                if "timestamp" in df.columns:
                    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
                elif isinstance(df.index, pd.DatetimeIndex):
                    df = df[(df.index >= start) & (df.index <= end)]

            # 5) Skip empty
            if df is None or df.empty:
                logger.warning("Skipping empty DataFrame: %s", file_path)
                continue

            logger.info("Loaded OHLC frame: %s | cols=%s | rows=%d", file_path, list(df.columns)[:10], len(df))
            frames.append(df)

    return frames
