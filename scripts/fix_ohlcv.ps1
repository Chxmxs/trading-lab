param(
  [Parameter(Mandatory=$true)][string]$Name,
  [string]$Registry = "configs/data_registry.yaml",
  [string]$OutDir = "data/cleaned"
)

# Ensure output folder exists
if (!(Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

# Run the Python fixer
python - <<'PYCODE' $Name $Registry $OutDir
import pandas as pd, numpy as np
from pathlib import Path
from adapters.data_adapters import Registry, validate_ohlcv_schema, to_pandas_freq, _read_csv, resample_ohlcv, audit_bars

name, registry_file, outdir = ARGV[1], ARGV[2], ARGV[3]
reg = Registry.from_yaml(registry_file)
kind, key = reg.resolve(name)
assert kind=="ohlcv", f"{name} is not OHLCV"
item = reg.ohlcv[key]
tf = to_pandas_freq(key.split("@")[1])
df = validate_ohlcv_schema(_read_csv(item))

# Sort and deduplicate
df = df.drop_duplicates("date").sort_values("date")

# Reindex to full timeline
start, end = df["date"].iloc[0], df["date"].iloc[-1]
full = pd.date_range(start=start.floor(tf), end=end.ceil(tf), freq=tf, tz="UTC")
df = df.set_index("date").reindex(full)

# Keep columns
df = df.rename_axis("date").reset_index()

out_path = Path(outdir)/f"{key.replace(':','_')}.csv"
df.to_csv(out_path, index=False)
print(f"[CLEANED] -> {out_path}")
rep = audit_bars(df, tf)
print("Post-fix audit:", rep)
PYCODE
