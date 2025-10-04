import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from adapters.data_adapters import (
    Registry,
    validate_ohlcv_schema,
    to_pandas_freq,
    _read_csv,
    resample_ohlcv,
    audit_bars,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="ohlcv:<SYMBOL>@<tf>")
    ap.add_argument("--registry", default="configs/data_registry.yaml")
    ap.add_argument("--outdir", default="data/cleaned")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    reg = Registry.from_yaml(args.registry)
    kind, key = reg.resolve(args.name)
    if kind != "ohlcv":
        raise SystemExit(f"{args.name} is not OHLCV")

    item = reg.ohlcv[key]
    tf = to_pandas_freq(key.split("@")[1])

    # 1) Load raw + basic schema
    raw = _read_csv(item)
    df = validate_ohlcv_schema(raw)

    # 2) Snap to grid (right-closed, right-labeled) by resampling to SAME tf
    #    This handles slight timestamp drift like xx:04:59 or xx:05:01.
    df = resample_ohlcv(df, tf)

    # 3) Deduplicate and sort just in case
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    # 4) Reindex to full timeline to expose gaps as empty rows
    start = df["date"].iloc[0].floor(tf)
    end   = df["date"].iloc[-1].ceil(tf)
    full = pd.date_range(start, end, freq=tf, tz="UTC")
    df = (df.set_index("date")
            .reindex(full)
            .rename_axis("date")
            .reset_index())

    # 5) Post-fix audit
    rep = audit_bars(
        df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"}),
        tf
    )
    print(f"[POST-FIX AUDIT] {key}")
    print(f"  duplicates:   {rep.duplicates}")
    print(f"  missing_bars:  {rep.missing_bars}")
    print(f"  misaligned:    {rep.misaligned}")
    print(f"  non_monotonic: {rep.non_monotonic}")

    # 6) Save cleaned CSV
    out_path = outdir / f"ohlcv_{key.replace(':','_')}.csv"
    df.to_csv(out_path, index=False)
    print(f"[CLEANED] -> {out_path}")

if __name__ == "__main__":
    main()
