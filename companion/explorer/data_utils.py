import pandas as pd
from pathlib import Path

def load_data_dirs(directories, date_range=None):
    frames = []
    for d in directories:
        path = Path(d)
        for file_path in path.glob("**/*"):
            if file_path.suffix.lower() in (".csv", ".parquet"):
                df = pd.read_csv(file_path) if file_path.suffix.lower() == ".csv" else pd.read_parquet(file_path)
                if date_range:
                    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
                frames.append(df)
    return frames
