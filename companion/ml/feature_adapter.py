from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path("artifacts/features")

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    if "timestamp" in cols:
        cols = ["timestamp"] + [c for c in cols if c != "timestamp"]
        df = df[cols]
    return df

def save_features(df: pd.DataFrame, symbol: str, timeframe: str, name: str) -> Path:
    base = ROOT / symbol / timeframe
    base.mkdir(parents=True, exist_ok=True)
    df2 = _ensure_cols(df.copy())
    p = base / f"{name}.csv"
    df2.to_csv(p, index=False)
    return p

def load_features(symbol: str, timeframe: str, name: str) -> pd.DataFrame:
    p = ROOT / symbol / timeframe / f"{name}.csv"
    df = pd.read_csv(p)
    return _ensure_cols(df)

def list_features(symbol: str, timeframe: str):
    """
    Return a sorted list of feature names available under ROOT/<symbol>/<timeframe>.
    Names are returned without the .csv extension.
    """
    base = ROOT / symbol / timeframe
    out = []
    try:
        if base.exists():
            for p in sorted(base.glob("*.csv")):
                out.append(p.stem)
    except Exception:
        pass
    return out
