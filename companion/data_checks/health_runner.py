# -*- coding: utf-8 -*-
from __future__ import annotations
import hashlib, importlib, inspect, json
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional

CACHE_VERSION = 3   # bump to invalidate old cache

health_mod = importlib.import_module("companion.data_checks.health")

def _dataset_signature(sources: Iterable[str]) -> Dict[str, Any]:
    sig: Dict[str, Any] = {}
    for s in sorted(set([str(x) for x in sources if x])):
        p = Path(s)
        if p.exists():
            st = p.stat()
            sig[s] = {"mtime": str(int(st.st_mtime)), "size": int(st.st_size)}
        else:
            sig[s] = {"mtime": "MISSING", "size": -1}
    return sig

def _hash_dict(d: Dict[str, Any]) -> str:
    b = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

def _read_csv(path: Path):
    import pandas as pd
    return pd.read_csv(str(path))

def _normalize_ohlcv_df(df):
    import pandas as pd
    lower_map = {c.lower(): c for c in df.columns}
    # timestamp
    src = None
    for cand in ("timestamp","ts","date","datetime"):
        if cand in lower_map:
            src = lower_map[cand]; break
    if src is None:
        return None
    if src != "timestamp":
        df = df.rename(columns={src: "timestamp"})
    # OHLCV names
    def _pick(name: str):
        if name in lower_map: return lower_map[name]
        for k,v in lower_map.items():
            if k.startswith(name): return v
        return None
    col_map = {}
    for base in ("open","high","low","close","volume"):
        src_col = _pick(base)
        if src_col and src_col != base:
            col_map[src_col] = base
    if col_map:
        df = df.rename(columns=col_map)
    for base in ("open","high","low","close","volume"):
        if base not in df.columns:
            df[base] = 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df[["timestamp","open","high","low","close","volume"]]

def _build_normalized_inputs(sources: Iterable[str], cache_dir: Path, cache_key: str):
    """Return (norm_csv_path or None, df_norm or None)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    src = None
    for s in sources:
        p = Path(str(s))
        if p.exists() and p.suffix.lower() in (".csv",".txt"):
            src = p; break
    if src is None:
        return (None, None)
    try:
        import pandas as pd
        df = _read_csv(src)
        df2 = _normalize_ohlcv_df(df)
        if df2 is None:
            return (None, None)
        out = cache_dir / f"normalized_{cache_key}.csv"
        df2.to_csv(out, index=False)
        return (out, df2)
    except Exception:
        return (None, None)

def _flex_call(fn, **kws):
    import pandas as pd
    sig = inspect.signature(fn)
    args: Dict[str, Any] = {}
    params = set(sig.parameters.keys())
    if "symbol" in params and "symbol" in kws: args["symbol"] = kws["symbol"]
    if "timeframe" in params and "timeframe" in kws: args["timeframe"] = kws["timeframe"]
    if "sources" in params and "sources" in kws: args["sources"] = kws["sources"]
    if "paths" in params and "paths" in kws: args["paths"] = kws["paths"]
    if "window" in params and "window" in kws: args["window"] = kws["window"]
    if "start" in params and "start" in kws: args["start"] = kws["start"]
    if "end" in params and "end" in kws: args["end"] = kws["end"]
    if "df" in params and "df" not in args:
        args["df"] = kws.get("df", pd.DataFrame())
    return fn(**args)

def preflight_data_health(*, symbol: str, timeframe: str, window: Dict[str,str], sources: Iterable[str], cache_root: str = "artifacts/_cache/data_health") -> Dict[str, Any]:
    cache_dir = Path(cache_root); cache_dir.mkdir(parents=True, exist_ok=True)
    sig = {"cache_version": CACHE_VERSION, "symbol": symbol, "timeframe": timeframe, "window": window, "dataset": _dataset_signature(sources)}
    digest = _hash_dict(sig)
    cache_key = f"{symbol}__{timeframe}__{digest[:16]}"
    out_json = cache_dir / f"{cache_key}.json"
    if out_json.exists():
        with open(out_json, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached["cache_key"] = cache_key
        cached["summary_path"] = str(out_json)
        return cached

    norm_csv, df_norm = _build_normalized_inputs(sources, cache_dir, cache_key)
    norm_sources = [str(norm_csv)] + list(sources) if norm_csv else list(sources)

    summarize_health = getattr(health_mod, "summarize_health")
    try:
        raw = _flex_call(
            summarize_health,
            symbol=symbol,
            timeframe=timeframe,
            sources=norm_sources,
            paths=norm_sources,
            window=window,
            start=window.get("start"),
            end=window.get("end"),
            df=df_norm,  # <<< pass normalized df explicitly
        )
    except Exception as e:
        raw = {"status":"fail","errors":[f"health_runner exception: {e.__class__.__name__}: {e}"],"symbol":symbol,"timeframe":timeframe,"window":window,"sources":norm_sources}

    # normalize return
    if isinstance(raw, dict):
        summary = dict(raw)
    elif isinstance(raw, (list,tuple)):
        status0 = str(raw[0]).lower() if len(raw)>=1 else "fail"
        details = raw[1] if len(raw)>=2 and isinstance(raw[1], dict) else {}
        summary = {"status": status0, **details}
    elif isinstance(raw, str):
        summary = {"status": raw.lower()}
    else:
        summary = {"status":"fail","errors":[f"unsupported summarize_health return type: {type(raw).__name__}"]}

    status = str(summary.get("status","")).lower().strip()
    if status not in ("pass","warn","fail"):
        status = "fail" if summary.get("errors") else "warn" if summary.get("warnings") else "pass"
    summary["status"] = status

    payload = {"status": status, "summary": summary, "symbol": symbol, "timeframe": timeframe, "window": window, "cache_key": cache_key}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    payload["summary_path"] = str(out_json)
    return payload