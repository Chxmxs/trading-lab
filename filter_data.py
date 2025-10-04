import pandas as pd
from pathlib import Path
import shutil

# Minimum number of rows required (set to 3 if n_splits=2)
min_rows = 3

# Directories containing your data
source_dirs = [
    Path("./data/parquet"),
    Path("./data/cleaned"),
    Path("./data/metrics")
]

# Directory to store files with enough rows
target_dir = Path("./data/usable")
target_dir.mkdir(exist_ok=True)

for source_dir in source_dirs:
    for file_path in source_dir.rglob("*"):
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            continue
        if len(df) >= min_rows:
            shutil.copy(file_path, target_dir / file_path.name)
