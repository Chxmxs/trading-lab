import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd

from adapters.data_adapters import (
    Registry,
    to_pandas_freq,
    _read_csv,
    validate_ohlcv_schema,
    resample_ohlcv,
    audit_bars,
)

def audit_one(name: str, reg: Registry):
    kind, key = reg.resolve(name)
    assert kind == "ohlcv"
    item = reg.ohlcv[key]
    tf = to_pandas_freq(key.split("@")[1])

    # Load raw + basic schema (no resampling unless file is finer than TF)
    raw = _read_csv(item)
    df = validate_ohlcv_schema(raw, allow_nans=True)

    # If file is finer than requested TF, roll up (safe)
    from pandas.tseries.frequencies import to_offset
    file_freq = tf  # assume equal; infer from filename if you like
    # (We won’t infer here; the per-file audit is about the target TF)
    df2 = resample_ohlcv(df, tf)

    rep = audit_bars(df2, tf)
    return {
        "name": key,
        "path": item.path,
        "tf": tf,
        "duplicates": rep.duplicates,
        "missing_bars": rep.missing_bars,
        "misaligned": rep.misaligned,
        "non_monotonic": rep.non_monotonic,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="configs/data_registry.yaml")
    ap.add_argument("--outdir", default="logs")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    reg = Registry.from_yaml(args.registry)

    rows = []
    for key in sorted(reg.ohlcv.keys(), key=str.lower):
        name = f"ohlcv:{key}"
        try:
            res = audit_one(name, reg)
        except Exception as e:
            rows.append({
                "name": key,
                "path": reg.ohlcv[key].path,
                "tf": key.split("@")[1],
                "duplicates": "ERR",
                "missing_bars": "ERR",
                "misaligned": "ERR",
                "non_monotonic": "ERR",
                "error": str(e)[:300],
            })
        else:
            res["error"] = ""
            rows.append(res)

    df = pd.DataFrame(rows, columns=[
        "name","path","tf","duplicates","missing_bars","misaligned","non_monotonic","error"
    ])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = outdir / f"ohlcv_audit_{stamp}.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False, max_colwidth=80))
    print(f"\nSaved CSV report -> {csv_path}")

if __name__ == "__main__":
    main()

