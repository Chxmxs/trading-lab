import argparse
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
import yaml

from adapters.data_adapters import (
    Registry,
    Item,
    to_pandas_freq,
    _infer_timeframe_from_filename,
    validate_ohlcv_schema,
    validate_metric_schema,
    resample_ohlcv,
    audit_bars,
)

# ---------- Helpers

def slugify_metric_name(filename: str) -> Optional[str]:
    """
    Build a metric slug from filename.
    Rules:
      - Leading 'Bitcoin ' -> btc., 'Ethereum ' -> eth.
      - Lowercase, spaces -> '_', remove punctuation.
      - Append qualifiers: (Mean)->_mean, (Median)->_median, (Total)->_total, (USD)->_usd, (MA7)->_ma7
      - Daily metrics -> @1d
    Returns 'btc.xxx' / 'eth.xxx' or None if not a known family.
    """
    name = Path(filename).stem  # no extension
    asset = None
    if name.startswith("Bitcoin "):
        asset = "btc"; n = name[len("Bitcoin "):]
    elif name.startswith("Ethereum "):
        asset = "eth"; n = name[len("Ethereum "):]
    else:
        return None

    # strip trailing ' - Day' if present
    n = re.sub(r"\s*-\s*Day$", "", n)

    # mark qualifiers to preserve during punctuation removal
    rep = {
        "(Mean)": "$mean", "(Median)": "$median", "(Total)": "$total",
        "(USD)": "$usd", "(MA7)": "$ma7"
    }
    for k,v in rep.items():
        n = n.replace(k, v)
    # basic normals
    n = n.replace("&","and")
    n = re.sub(r"[(),]", "", n)
    n = re.sub(r"\s+and\s+", "_", n)
    n = re.sub(r"\s+", "_", n)
    n = n.lower()
    # restore markers
    n = n.replace("$mean","_mean").replace("$median","_median").replace("$total","_total").replace("$usd","_usd").replace("$ma7","_ma7")
    return f"{asset}.{n}"

def is_ohlcv_csv(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    needed = {"open","high","low","close"}
    has_ts = bool({"timestamp","date","datetime","time"} & cols)
    return has_ts and needed.issubset(cols)

def detect_timeframe_from_name(path: Path) -> Optional[str]:
    inf = _infer_timeframe_from_filename(str(path))
    if inf:
        # normalize to our pandas freq
        return to_pandas_freq(inf)
    return None

def clean_ohlcv_df(df_raw: pd.DataFrame, tf: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # relaxed schema (allow NaNs, we’ll align and reindex)
    df = validate_ohlcv_schema(df_raw, allow_nans=True)
    # snap to grid at same TF (right-closed/right-labeled)
    df = resample_ohlcv(df, tf)
    # drop dupes, sort
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    # reindex to expose gaps
    start = df["date"].iloc[0]
    end   = df["date"].iloc[-1]
    full = pd.date_range(start.floor(tf), end.ceil(tf), freq=tf, tz="UTC")
    df = (df.set_index("date").reindex(full).rename_axis("date").reset_index())
    rep = audit_bars(
        df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"}),
        tf
    )
    stats = {
        "duplicates": rep.duplicates,
        "missing_bars": rep.missing_bars,
        "misaligned": rep.misaligned,
        "non_monotonic": rep.non_monotonic,
        "rows": len(df),
        "first": df["date"].min(),
        "last": df["date"].max(),
    }
    return df, stats

def clean_metric_df(df_raw: pd.DataFrame, value_column: Optional[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Auto-detect inside validate_metric_schema if value_column not present
    df = validate_metric_schema(df_raw, value_column)
    rep = audit_bars(
        df.rename(columns={"value":"close"}).assign(open=np.nan, high=np.nan, low=np.nan, volume=np.nan)[
            ["date","open","high","low","close","volume"]
        ],
        "1D",
    )
    stats = {
        "duplicates": rep.duplicates,
        "missing_bars": rep.missing_bars,
        "misaligned": rep.misaligned,
        "non_monotonic": rep.non_monotonic,
        "rows": len(df),
        "first": df["date"].min(),
        "last": df["date"].max(),
    }
    return df, stats

def write_outputs(df: pd.DataFrame, out_csv: Path, out_parq: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_parq.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_parq, index=False)  # pyarrow / fastparquet
    except Exception:
        pass

def ensure_registry_entry(reg_file: Path, kind: str, key: str, path_str: str, value_column: Optional[str] = None):
    with open(reg_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "ohlcv" not in data: data["ohlcv"] = {}
    if "metric" not in data: data["metric"] = {}

    target = data["ohlcv"] if kind == "ohlcv" else data["metric"]
    if key not in target:
        entry = {"path": path_str, "tz": "UTC"}
        if kind == "metric" and value_column:
            entry["value_column"] = value_column
        target[key] = entry
    else:
        # update just the path (don’t clobber value_column if user set it)
        target[key]["path"] = path_str

    with open(reg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

# ---------- Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Root folder containing your CSVs")
    ap.add_argument("--registry", default="configs/data_registry.yaml")
    ap.add_argument("--out-clean", default="data/cleaned")
    ap.add_argument("--out-parquet", default="data/parquet")
    ap.add_argument("--update-registry", action="store_true", help="Append/update entries in data_registry.yaml")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    reg_file = Path(args.registry)
    out_clean = Path(args.out_clean)
    out_parq = Path(args.out_parquet)

    rows = []
    for path in data_dir.rglob("*.csv"):
        try:
            df_raw = pd.read_csv(path)
            cols_lower = {c.lower() for c in df_raw.columns}
            # Decide OHLCV vs Metric
            if is_ohlcv_csv(df_raw):
                # determine target tf
                tf = detect_timeframe_from_name(path)
                if not tf:
                    # fallback: 1min if nothing else; you can change default here
                    tf = to_pandas_freq("1m")
                df_clean, stats = clean_ohlcv_df(df_raw, tf)

                # Build key: SYMBOL@tf
                # Try to extract symbol from filename like BTCUSD_5.csv -> BTCUSD
                m = re.match(r"([A-Za-z0-9]+)[-_]?\d{1,5}$", path.stem)
                symbol = m.group(1).upper() if m else path.stem.upper()
                # Normalize tf for key: we want short codes where possible
                tf_key = path.stem.split("_")[-1]
                try:
                    # make short code like 5m / 60m / 1d from pandas freq
                    if tf.endswith("min"):
                        tf_short = f"{int(tf[:-3])}m"
                    elif tf.upper().endswith("D"):
                        dnum = int(tf[:-1]) if tf[:-1] else 1
                        tf_short = "1d" if dnum == 1 else f"{dnum}d"
                    elif tf.upper() == "1W":
                        tf_short = "1w"
                    else:
                        tf_short = tf
                except Exception:
                    tf_short = tf

                key = f"{symbol}@{tf_short}"
                out_csv = out_clean / f"ohlcv_{symbol}@{tf_short}.csv"
                out_parquet = out_parq / f"ohlcv_{symbol}@{tf_short}.parquet"
                write_outputs(df_clean, out_csv, out_parquet)

                if args.update_registry:
                    ensure_registry_entry(reg_file, "ohlcv", key, str(out_csv.as_posix()))

                rows.append({
                    "kind":"ohlcv", "key": key, "src": str(path), "clean_csv": str(out_csv),
                    "rows": stats["rows"], "first": stats["first"], "last": stats["last"],
                    "duplicates": stats["duplicates"], "missing_bars": stats["missing_bars"],
                    "misaligned": stats["misaligned"], "non_monotonic": stats["non_monotonic"], "error":""
                })

            else:
                # Treat as metric if it has a time column and at least one other column
                if not ({"timestamp","date","datetime","time"} & cols_lower):
                    # skip unknown file, not metric nor ohlcv
                    rows.append({
                        "kind":"unknown","key":"", "src": str(path), "clean_csv":"", "rows":0,
                        "first":"", "last":"", "duplicates":"","missing_bars":"",
                        "misaligned":"", "non_monotonic":"", "error":"No timestamp & not OHLCV"
                    })
                    continue

                # Derive metric slug from filename (btc./eth.)
                slug = slugify_metric_name(path.name)
                if not slug:
                    # leave unregistered but still clean: try auto-detect value column
                    slug = f"metric.{path.stem.lower()}"
                df_clean, stats = clean_metric_df(df_raw, value_column=None)

                key = f"{slug}@1d"
                out_csv = out_clean / f"metric_{slug}@1d.csv"
                out_parquet = out_parq / f"metric_{slug}@1d.parquet"
                write_outputs(df_clean, out_csv, out_parquet)

                if args.update_registry:
                    ensure_registry_entry(reg_file, "metric", f"{slug}@1d", str(out_csv.as_posix()))
                rows.append({
                    "kind":"metric", "key": f"{slug}@1d", "src": str(path), "clean_csv": str(out_csv),
                    "rows": stats["rows"], "first": stats["first"], "last": stats["last"],
                    "duplicates": stats["duplicates"], "missing_bars": stats["missing_bars"],
                    "misaligned": stats["misaligned"], "non_monotonic": stats["non_monotonic"], "error":""
                })

        except Exception as e:
            rows.append({
                "kind":"error", "key":"", "src": str(path), "clean_csv":"",
                "rows":"", "first":"", "last":"", "duplicates":"", "missing_bars":"",
                "misaligned":"", "non_monotonic":"", "error": str(e)[:300]
            })

    # write report
    Path("logs").mkdir(parents=True, exist_ok=True)
    report = Path("logs") / "prepare_report.csv"
    pd.DataFrame(rows).to_csv(report, index=False)
    print(f"Saved report -> {report}")
    # quick console preview of issues
    issues = [r for r in rows if (str(r.get("error","")) or "").strip()]
    if issues:
        print("\n=== Issues Detected (first 10) ===")
        for r in issues[:10]:
            print(f"- {r['src']}: {r['error']}")

if __name__ == "__main__":
    main()
