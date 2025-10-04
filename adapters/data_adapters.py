from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dateutil import tz
from pydantic import BaseModel, Field, field_validator

import pybroker
from pybroker.data import DataSource


# -------------------- Timeframe helpers --------------------

_PANDAS_FREQ_ALIASES = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "60min",
    "1h": "60min", "2h": "120min", "4h": "240min",
    "1d": "1D", "1w": "1W",
}

def to_pandas_freq(tf: str) -> str:
    """
    Normalizes many common time frame spellings to a pandas offset string.
    Accepts: "5m", "240m", "720m", "5min", "5T", "60m", "1h", "1H", "1d", "1D", "1w", "1W",
              or bare digits ("5" => "5min").
    """
    tf = str(tf).strip()
    tf_l = tf.lower()

    # Digits => minutes
    if tf_l.isdigit():
        return f"{int(tf_l)}min"

    # Exact short aliases
    aliases = {
        "1m":"1min","5m":"5min","15m":"15min","30m":"30min","60m":"60min",
        "1h":"60min","2h":"120min","4h":"240min",
        "1d":"1D","1w":"1W",
    }
    if tf_l in aliases:
        return aliases[tf_l]

    # Generic minute styles:
    # "5min" / "15min" / "240min"
    if tf_l.endswith("min") and tf_l[:-3].isdigit():
        return f"{int(tf_l[:-3])}min"
    # "5t" / "15t" (pandas alias)
    if tf_l.endswith("t") and tf_l[:-1].isdigit():
        return f"{int(tf_l[:-1])}min"
    # "240m" / "720m" etc.
    if tf_l.endswith("m") and tf_l[:-1].isdigit():
        return f"{int(tf_l[:-1])}min"

    # Hours: "1h","2h","4h","12h", etc.
    if tf_l.endswith("h") and tf_l[:-1].isdigit():
        return f"{int(tf_l[:-1]) * 60}min"

    # Days / Weeks (accept "1d"/"2d" and "1w")
    if tf_l.endswith("d") and tf_l[:-1].isdigit():
        d = int(tf_l[:-1])
        return "1D" if d == 1 else f"{d}D"
    if tf_l.endswith("w"):
        return "1W"  # pandas supports whole weeks only

    # Already valid pandas code like "1D"/"1W"
    if tf in {"1D","1W"} or tf_l in {"1d","1w"}:
        return tf if tf in {"1D","1W"} else tf.upper()

    raise ValueError(f"Unrecognized timeframe: {tf}")
def _infer_timeframe_from_filename(path: str) -> Optional[str]:
    m = re.search(r"_(\d{1,5})(?=\.[cC][sS][vV]$)", Path(path).name)
    if not m:
        return None
    mins = int(m.group(1))
    if mins % 1440 == 0:
        days = mins // 1440
        return "1D" if days == 1 else f"{days}D"
    return f"{mins}min"

def _colmap_lower(df: pd.DataFrame) -> Dict[str, str]:
    return {c.lower(): c for c in df.columns}

def _find_col(colmap: Dict[str, str], options: Iterable[str]) -> Optional[str]:
    for opt in options:
        if opt in colmap:
            return colmap[opt]
    return None


# -------------------- Registry --------------------

class Item(BaseModel):
    path: str
    tz: str = "UTC"
    value_column: Optional[str] = None  # for metric CSVs

    @field_validator("tz")
    @classmethod
    def _tz_ok(cls, v: str) -> str:
        return v

class Registry(BaseModel):
    ohlcv: Dict[str, Item] = Field(default_factory=dict)
    metric: Dict[str, Item] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, file: str | Path) -> "Registry":
        with open(file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        oh = {}
        for k, v in (data.get("ohlcv") or {}).items():
            oh[k.strip()] = v
        me = {}
        for k, v in (data.get("metric") or {}).items():
            me[k.strip()] = v
        return cls(ohlcv=oh, metric=me)

    def resolve(self, name: str) -> Tuple[str, str]:
        kind, slug, tf = _parse_name(name)
        key = f"{slug}@{tf}"
        pool = self.ohlcv if kind == "ohlcv" else self.metric
        if key not in pool:
            raise KeyError(f"{kind}:{key} not found in registry. Add it to data_registry.yaml.")
        return kind, key


# -------------------- Name parsing --------------------

_NAME_RE = re.compile(r"^(ohlcv|metric):([a-z0-9._-]+)@(\d+[mhdw])$", re.IGNORECASE)

@dataclass(frozen=True)
class ParsedName:
    kind: str
    slug: str
    tf: str

def _parse_name(name: str) -> Tuple[str, str, str]:
    n = str(name).strip()
    m = _NAME_RE.match(n)
    if not m:
        raise ValueError("Invalid data name. Use 'ohlcv:SYMBOL@tf' or 'metric:slug@tf'.")
    kind = m.group(1).lower()
    slug = m.group(2)
    tf = m.group(3).lower()
    return kind, slug, tf


# -------------------- Validation & auditing --------------------

@dataclass
class AuditReport:
    duplicates: int
    missing_bars: int
    gaps: List[Tuple[pd.Timestamp, pd.Timestamp, int]]
    misaligned: int
    non_monotonic: bool
    tz: str

def _ensure_dt_utc(df: pd.DataFrame, ts_col: str) -> pd.Series:
    s = df[ts_col]

    # If numeric-like, treat as Unix epoch (seconds or milliseconds)
    is_numeric_like = (
        pd.api.types.is_integer_dtype(s)
        or pd.api.types.is_float_dtype(s)
        or s.astype(str).str.fullmatch(r"\d+").all()
    )

    if is_numeric_like:
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.isna().all():
            bad = s.head(3).tolist()
            raise ValueError(f"Unparseable numeric timestamps (first 3): {bad}")
        # Heuristic: ms if median > 1e11, else seconds
        med = s_num.dropna().median()
        unit = "ms" if med > 1e11 else "s"
        ts = pd.to_datetime(s_num, unit=unit, utc=True)
    else:
        # String-like – let pandas parse, then normalize to UTC
        ts = pd.to_datetime(s, utc=False, errors="coerce")
        if ts.isna().any():
            bad = s[ts.isna()].head(3).tolist()
            raise ValueError(f"Unparseable timestamps (first 3): {bad}")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")

    return ts
def validate_ohlcv_schema(df: pd.DataFrame, allow_nans: bool = False) -> pd.DataFrame:
    colmap = _colmap_lower(df)
    ts_col = _find_col(colmap, ["timestamp", "date", "datetime", "time"])
    o = _find_col(colmap, ["open"]); h = _find_col(colmap, ["high"])
    l = _find_col(colmap, ["low"]);  c = _find_col(colmap, ["close"])
    v = _find_col(colmap, ["volume", "vol"])
    if not all([ts_col, o, h, l, c]):
        raise ValueError("Missing one of required columns: timestamp/open/high/low/close")
    out = pd.DataFrame({
        "date": _ensure_dt_utc(df, ts_col),
        "open": pd.to_numeric(df[o], errors="coerce"),
        "high": pd.to_numeric(df[h], errors="coerce"),
        "low":  pd.to_numeric(df[l], errors="coerce"),
        "close":pd.to_numeric(df[c], errors="coerce"),
    })
    out["volume"] = pd.to_numeric(df[v], errors="coerce") if v else np.nan
    if not allow_nans and out[["open","high","low","close"]].isna().any().any(): raise ValueError("NaNs detected in O/H/L/C after coercion.")
    return out

def _auto_detect_value_column(df: pd.DataFrame, ts_col: str) -> Optional[str]:
    """
    Try to find the primary numeric series column in a metric CSV.
    Preference order:
      1) column literally named 'value' (case-insensitive)
      2) names containing 'value', 'price', 'count', 'total', 'mean', 'median', 'usd'
      3) the column with the highest fraction of numeric, non-null values
    """
    candidates = [c for c in df.columns if c != ts_col]
    if not candidates:
        return None

    # 1) exact 'value'
    for c in candidates:
        if c.strip().lower() == "value":
            return c

    # 2) name-based preference
    pref_keywords = ("value", "price", "count", "total", "mean", "median", "usd")
    name_scored = []
    for c in candidates:
        cl = c.lower()
        score = sum(k in cl for k in pref_keywords)
        name_scored.append((score, c))
    name_scored.sort(reverse=True)  # highest score first

    # 3) numeric density
    def numeric_ok(col) -> float:
        s = pd.to_numeric(df[col], errors="coerce")
        return float((~s.isna()).mean())

    best_col = None
    best_density = -1.0
    for _, c in name_scored:
        dens = numeric_ok(c)
        if dens > best_density:
            best_density = dens
            best_col = c

    # require at least some numeric coverage
    if best_density <= 0.2:
        return None
    return best_col

def validate_metric_schema(df: pd.DataFrame, value_column: Optional[str]) -> pd.DataFrame:
    colmap = _colmap_lower(df)
    ts_col = _find_col(colmap, ["timestamp","date","datetime","time"])
    if not ts_col:
        raise ValueError("Metric CSV must have a timestamp/date column.")

    chosen = None
    if value_column and value_column in df.columns:
        chosen = value_column
    else:
        # auto-detect if missing/wrong
        chosen = _auto_detect_value_column(df, ts_col=colmap[ts_col] if ts_col in colmap else ts_col)
        if chosen is None:
            raise ValueError(
                f"Metric value column not found. Available columns: {list(df.columns)}"
            )

    out = pd.DataFrame({
        "date": _ensure_dt_utc(df, colmap[ts_col] if ts_col in colmap else ts_col),
        "value": pd.to_numeric(df[chosen], errors="coerce")
    }).dropna(subset=["value"])
    return out

def audit_bars(df: pd.DataFrame, pandas_freq: str) -> AuditReport:
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    non_monotonic = not df["date"].is_monotonic_increasing
    dups = int(df["date"].duplicated().sum())
    rule = pd.tseries.frequencies.to_offset(pandas_freq)
    misaligned = int((df["date"].dt.second != 0).sum() + (df["date"].dt.microsecond != 0).sum())
    if hasattr(rule, "n") and rule.n % 60 == 0:
        step_min = rule.n // 60
        misaligned += int(((df["date"].dt.minute % step_min) != 0).sum())
    start = df["date"].iloc[0].floor(pandas_freq)
    end = df["date"].iloc[-1].ceil(pandas_freq)
    full = pd.date_range(start, end, freq=pandas_freq, tz="UTC")
    missing_idx = full.difference(df["date"])
    missing_bars = len(missing_idx)
    gaps = []
    if len(df) > 1:
        deltas = df["date"].diff().dropna()
        expected = pd.to_timedelta(rule)
        gap_locs = np.where(deltas > expected)[0]
        for i in gap_locs:
            prev = df["date"].iloc[i-1]; curr = df["date"].iloc[i]
            miss = int((curr - prev) / expected) - 1
            gaps.append((prev, curr, miss))
    return AuditReport(duplicates=dups, missing_bars=missing_bars, gaps=gaps,
                       misaligned=misaligned, non_monotonic=non_monotonic, tz="UTC")


# -------------------- Resampling (no forward leaks) --------------------

def resample_ohlcv(df: pd.DataFrame, pandas_freq: str) -> pd.DataFrame:
    g = (df.set_index("date")
           .resample(pandas_freq, label="right", closed="right"))
    agg = {
        "open": "first",
        "high": "max",
        "low":  "min",
        "close":"last",
        "volume":"sum"
    }
    out = g.agg(agg).dropna(subset=["open","high","low","close"]).reset_index()
    return out


# -------------------- Feature merging (as-of, backward) --------------------

def asof_join_features(ohlcv: pd.DataFrame, features: Dict[str, pd.DataFrame],
                       tolerance: Optional[pd.Timedelta] = None) -> pd.DataFrame:
    out = ohlcv.copy()
    for name, feat in features.items():
        feat = feat.sort_values("date")
        tol = tolerance if tolerance is not None else pd.Timedelta("2D")
        out = pd.merge_asof(
            out.sort_values("date"),
            feat[["date","value"]].rename(columns={"value": name}).sort_values("date"),
            on="date", direction="backward", tolerance=tol
        ).sort_values("date")
    return out


# -------------------- CSV loaders --------------------

def _read_csv(item: Item) -> pd.DataFrame:
    return pd.read_csv(item.path)

def load_ohlcv_item(item: Item, requested_freq: str) -> Tuple[pd.DataFrame, str]:
    raw = _read_csv(item)
    df = validate_ohlcv_schema(raw, allow_nans=True)

    # Determine file frequency (fallback to requested)
    file_freq = _infer_timeframe_from_filename(item.path) or "infer"
    if file_freq == "infer":
        file_freq = requested_freq

    # Only roll up from finer -> coarser. Never upsample.
    if pd.tseries.frequencies.to_offset(file_freq) < pd.tseries.frequencies.to_offset(requested_freq):
        df = resample_ohlcv(df, requested_freq)
    elif pd.tseries.frequencies.to_offset(file_freq) > pd.tseries.frequencies.to_offset(requested_freq):
        raise ValueError(f"Cannot upsample from {file_freq} to {requested_freq} without look-ahead.")

    # Audit after alignment
    rep = audit_bars(df, requested_freq)

    # Hard-stop on structural issues
    hard_fail = []
    if rep.duplicates:
        hard_fail.append(f"duplicates={rep.duplicates}")
    if rep.misaligned:
        hard_fail.append(f"misaligned={rep.misaligned}")
    if rep.non_monotonic:
        hard_fail.append("non_monotonic=True")

    if hard_fail:
        raise ValueError("OHLCV audit failed: " + ", ".join(hard_fail))

    # Missing bars are allowed (markets/exchanges sometimes have holes).
    # We just warn so you can decide to fill later.
    if rep.missing_bars:
        try:
            print(f"[WARN] OHLCV missing bars: {rep.missing_bars} in {item.path}")
        except Exception:
            pass

    return df, requested_freq
def load_metric_item(item: Item) -> pd.DataFrame:
    raw = _read_csv(item)
    df = validate_metric_schema(raw, item.value_column)
    rep = audit_bars(
        df.rename(columns={"value":"close"}).assign(open=np.nan, high=np.nan, low=np.nan, volume=np.nan)[
            ["date","open","high","low","close","volume"]
        ],
        "1D",
    )
    if rep.duplicates or rep.non_monotonic:
        raise ValueError("Metric audit failed (duplicates or non-monotonic).")
    return df


# -------------------- PyBroker DataSource --------------------

class CSVFolderDataSource(DataSource):
    """
    A PyBroker DataSource backed by our YAML registry.
    Usage: ds = CSVFolderDataSource('configs/data_registry.yaml')
           df = ds.query(['BTCUSD'], start_date='2017-01-01', end_date='2025-03-14', timeframe='5m')
    """
    def __init__(self, registry_file: str | Path):
        super().__init__()
        self.registry = Registry.from_yaml(registry_file)

    def _fetch_data(
        self,
        symbols: Iterable[str],
        start_date: str | datetime | pd.Timestamp,
        end_date: str | datetime | pd.Timestamp,
        timeframe: str,
        adjust=None,
        **kwargs,
    ) -> pd.DataFrame:
        # We ignore "adjust" because we provide already-adjusted bars from CSV.
        tf = to_pandas_freq(str(timeframe).lower())
        frames = []
        for sym in symbols:
            key = f"{sym}@{str(timeframe).lower()}"
            if key not in self.registry.ohlcv:
                candidates = [k for k in self.registry.ohlcv.keys()
                              if k.split("@", 1)[0].lower() == sym.lower()]
                if not candidates:
                    raise KeyError(f"No OHLCV entry for symbol {sym} in registry.")
                def _off(k): return pd.tseries.frequencies.to_offset(
                    to_pandas_freq(k.split("@")[1])
                )
                cand = sorted(candidates, key=lambda k: _off(k))
                chosen = cand[0]
                item = self.registry.ohlcv[chosen]
            else:
                item = self.registry.ohlcv[key]

            df, _ = load_ohlcv_item(item, tf)
            s = pd.to_datetime(start_date, utc=True)
            e = pd.to_datetime(end_date, utc=True)
            df = df[(df["date"] >= s) & (df["date"] <= e)].copy()
            df.insert(0, "symbol", sym)
            frames.append(df)

        out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
        # Register extras so strategies can access them
        pybroker.register_columns(["volume"])
        return out
# -------------------- CLI helpers --------------------

def print_audit(report: AuditReport, label: str):
    print(f"[AUDIT] {label}")
    print(f"  duplicates:   {report.duplicates}")
    print(f"  missing_bars:  {report.missing_bars}")
    print(f"  misaligned:    {report.misaligned}")
    print(f"  non_monotonic: {report.non_monotonic}")
    if report.gaps:
        print("  gaps:")
        for (a,b,miss) in report.gaps[:20]:
            print(f"    {a} -> {b}  (missing {miss})")
        if len(report.gaps) > 20:
            print(f"    ... {len(report.gaps)-20} more")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Audit / Load CSVs via registry.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_a = sub.add_parser("audit-ohlcv")
    p_a.add_argument("--name", required=True, help="e.g., ohlcv:BTCUSD@5m")
    p_a.add_argument("--registry", default="configs/data_registry.yaml")

    p_m = sub.add_parser("audit-metric")
    p_m.add_argument("--name", required=True, help="e.g., metric:btc.exchange_to_miner_total@1d")
    p_m.add_argument("--registry", default="configs/data_registry.yaml")

    p_r = sub.add_parser("resample")
    p_r.add_argument("--name", required=True, help="ohlcv:<...>")
    p_r.add_argument("--to", required=True, help="target tf like 60m or 1d")
    p_r.add_argument("--registry", default="configs/data_registry.yaml")
    p_r.add_argument("--out", required=False)

    p_q = sub.add_parser("query")
    p_q.add_argument("--symbols", nargs="+", required=True)
    p_q.add_argument("--start", required=True)
    p_q.add_argument("--end", required=True)
    p_q.add_argument("--tf", required=True)
    p_q.add_argument("--registry", default="configs/data_registry.yaml")

    args = p.parse_args()
    reg = Registry.from_yaml(args.registry)

    if args.cmd == "audit-ohlcv":
        kind, key = reg.resolve(args.name)
        assert kind == "ohlcv"
        item = reg.ohlcv[key]
        tf = to_pandas_freq(key.split("@")[1])
        df, _ = load_ohlcv_item(item, tf)
        rep = audit_bars(df, tf)
        print_audit(rep, key)

    elif args.cmd == "audit-metric":
        kind, key = reg.resolve(args.name)
        assert kind == "metric"
        item = reg.metric[key]
        df = load_metric_item(item)
        rep = audit_bars(
            df.rename(columns={"value":"close"}).assign(open=np.nan, high=np.nan, low=np.nan, volume=np.nan)[
                ["date","open","high","low","close","volume"]
            ],
            "1D",
        )
        print_audit(rep, key)

    elif args.cmd == "resample":
        kind, key = reg.resolve(args.name)
        assert kind == "ohlcv"
        item = reg.ohlcv[key]
        target = to_pandas_freq(args.to)
        srcdf = validate_ohlcv_schema(_read_csv(item))
        out = resample_ohlcv(srcdf, target)
        out_path = args.out or (Path(item.path).with_name(Path(item.path).stem + f"_{args.to}.csv"))
        out.to_csv(out_path, index=False)
        print(f"Resampled -> {out_path}")

    elif args.cmd == "query":
        ds = CSVFolderDataSource(args.registry)
        df = ds.query(args.symbols, start_date=args.start, end_date=args.end, timeframe=args.tf)
        print(df.head(10))
        print(df.tail(3))
        print(f"Rows: {len(df):,}")







