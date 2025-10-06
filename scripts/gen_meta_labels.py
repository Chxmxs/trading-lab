# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Batch label generation CLI (tradebot-mlfl env).
Reads a simple JSON config and writes Parquet labels per job.

Usage:
  python scripts/gen_meta_labels.py --config configs/ml_labels.json
  python scripts/gen_meta_labels.py --symbol BTCUSD --tf 15m --method triple --price_csv data/cleaned/ohlcv_BTCUSD@15m.csv
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# repo bootstrap
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from companion.ml.labels import generate_labels_to_parquet

def run_job(job: Dict[str, Any]) -> str:
    method = job.get("method", "triple")
    price_csv = job["price_csv"]
    symbol = job["symbol"]
    tf = job.get("timeframe") or job.get("tf")
    params = job.get("params") or {}
    out_root = job.get("out_root", "data/parquet")
    out_path = generate_labels_to_parquet(
        method=method,
        price_csv=price_csv,
        symbol=symbol,
        timeframe=tf,
        out_root=out_root,
        params=params,
    )
    print(f"[labels] {symbol}@{tf} -> {out_path}")
    return out_path

def main(argv: Optional[List[str]] = None):
    import argparse
    p = argparse.ArgumentParser("Meta-label generator")
    p.add_argument("--config", help="Path to JSON config with 'jobs':[...]")
    p.add_argument("--symbol")
    p.add_argument("--tf")
    p.add_argument("--method", default="triple", choices=["triple", "fixed"])
    p.add_argument("--price_csv")
    p.add_argument("--pt", type=float)
    p.add_argument("--sl", type=float)
    p.add_argument("--pt_mult", type=float)
    p.add_argument("--sl_mult", type=float)
    p.add_argument("--horizon_bars", type=int)
    p.add_argument("--out_root", default="data/parquet")
    p.add_argument("--side", default="long", choices=["long","short", "none"])
    args = p.parse_args(argv)

    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        for job in cfg.get("jobs", []):
            run_job(job)
        return

    # single job via flags
    if not (args.symbol and args.tf and args.price_csv):
        print("Provide --symbol, --tf, --price_csv OR --config.")
        sys.exit(2)

    params = {}
    if args.method == "fixed":
        if args.pt is not None: params["pt"] = args.pt
        if args.sl is not None: params["sl"] = args.sl
    else:
        if args.pt_mult is not None: params["pt_mult"] = args.pt_mult
        if args.sl_mult is not None: params["sl_mult"] = args.sl_mult
    if args.horizon_bars is not None: params["horizon_bars"] = args.horizon_bars
    if args.side and args.side != "none": params["side"] = args.side

    run_job({
        "method": args.method,
        "price_csv": args.price_csv,
        "symbol": args.symbol,
        "timeframe": args.tf,
        "out_root": args.out_root,
        "params": params
    })

if __name__ == "__main__":
    main()
