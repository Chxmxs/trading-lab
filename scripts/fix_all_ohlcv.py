import argparse
from pathlib import Path
import pandas as pd
import yaml

from adapters.data_adapters import (
    Registry,
    to_pandas_freq,
    _read_csv,
    validate_ohlcv_schema,
    resample_ohlcv,
    audit_bars,
)

def clean_one(name: str, reg: Registry, outdir: Path):
    kind, key = reg.resolve(name)
    assert kind == "ohlcv"
    item = reg.ohlcv[key]
    tf = to_pandas_freq(key.split("@")[1])

    raw = _read_csv(item)
    df = validate_ohlcv_schema(raw, allow_nans=True)

    # Snap to grid (right-closed/right-labeled) at same TF
    df = resample_ohlcv(df, tf)

    # Drop duplicates, sort
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    # Reindex to full grid (explicit gap rows)
    start = df["date"].iloc[0].floor(tf)
    end   = df["date"].iloc[-1].ceil(tf)
    full = pd.date_range(start, end, freq=tf, tz="UTC")
    df = (df.set_index("date").reindex(full).rename_axis("date").reset_index())

    # Audit post-fix
    rep = audit_bars(
        df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"}),
        tf
    )

    out_path = outdir / f"ohlcv_{key.replace(':','_')}.csv"
    df.to_csv(out_path, index=False)
    return str(out_path), rep

def update_registry_path(reg_file: Path, key: str, new_path: str):
    with open(reg_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "ohlcv" not in data:
        raise ValueError("No 'ohlcv' section in registry.")
    if key not in data["ohlcv"]:
        raise ValueError(f"Key {key} not found in registry.")
    data["ohlcv"][key]["path"] = new_path
    with open(reg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="configs/data_registry.yaml")
    ap.add_argument("--outdir", default="data/cleaned")
    ap.add_argument("--update-registry", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    reg_file = Path(args.registry)
    reg = Registry.from_yaml(reg_file)

    rows = []
    for key in sorted(reg.ohlcv.keys(), key=str.lower):
        name = f"ohlcv:{key}"
        try:
            out_path, rep = clean_one(name, reg, outdir)
            if args.update_registry:
                update_registry_path(reg_file, key, str(Path(out_path).as_posix()))
            rows.append({
                "name": key,
                "cleaned_path": str(Path(out_path).as_posix()),
                "duplicates": rep.duplicates,
                "missing_bars": rep.missing_bars,
                "misaligned": rep.misaligned,
                "non_monotonic": rep.non_monotonic,
            })
            print(f"[CLEANED] {key} -> {out_path} | misaligned={rep.misaligned}, missing={rep.missing_bars}")
        except Exception as e:
            rows.append({
                "name": key,
                "cleaned_path": "",
                "duplicates": "ERR",
                "missing_bars": "ERR",
                "misaligned": "ERR",
                "non_monotonic": "ERR",
                "error": str(e)[:300],
            })
            print(f"[ERROR] {key}: {e}")

    report = Path("logs") / "ohlcv_clean_report.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report, index=False)
    print(f"\nSaved clean report -> {report}")

if __name__ == "__main__":
    main()
