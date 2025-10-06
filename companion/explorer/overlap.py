from __future__ import annotations
from pathlib import Path
from typing import Union, Iterable, Any
import os
import json
import time
import pandas as pd

__all__ = ['load_trade_structure','interval_overlap_score','jaccard_points',
           'prune_overlap_strategies','load_master','prune_master_items','save_master']

_CANON = ['run_id','strategy','symbol','timeframe','side','qty',
          'entry_time','entry_price','exit_time','exit_price','pnl','trade_id','position_id']

_COL_MAP = {
    'run':'run_id','runid':'run_id','strategy_name':'strategy','tf':'timeframe',
    'ticker':'symbol','asset':'symbol','side_name':'side','direction':'side',
    'quantity':'qty','size':'qty','entry':'entry_time','entry_ts':'entry_time',
    'open_time':'entry_time','entryprice':'entry_price','open_price':'entry_price',
    'exit':'exit_time','exit_ts':'exit_time','close_time':'exit_time',
    'exitprice':'exit_price','close_price':'exit_price','profit':'pnl',
    'pnl_quote':'pnl','pnl_usd':'pnl','id':'trade_id','position':'position_id',
}

def _to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors='coerce', utc=True)

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        k = str(c).strip().lower()
        ren[c] = _COL_MAP.get(k, k)
    df = df.rename(columns=ren)
    if 'side' in df.columns:
        df['side'] = (
            df['side'].astype(str).str.upper().map({
                'BUY':'LONG','LONG':'LONG','L':'LONG',
                'SELL':'SHORT','SHORT':'SHORT','S':'SHORT'
            }).fillna(df['side'])
        )
    for t in ('entry_time','exit_time'):
        if t in df.columns:
            df[t] = _to_utc(df[t])
    defaults = {
        'run_id':'','strategy':'','symbol':'','timeframe':'',
        'side':'','qty':0.0,'entry_time':pd.NaT,'entry_price':float('nan'),
        'exit_time':pd.NaT,'exit_price':float('nan'),'pnl':0.0,
        'trade_id':'','position_id':'',
    }
    for k,v in defaults.items():
        if k not in df.columns:
            df[k] = v
    extras = [c for c in df.columns if c not in _CANON]
    df = df[_CANON + extras]
    if 'entry_time' in df.columns:
        df = df.sort_values('entry_time', kind='stable', na_position='last')
    return df

def load_trade_structure(source: Union[str, os.PathLike, pd.DataFrame], *, assume_csv_has_header: bool = True) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Trades file not found: {p}")
        kw = {}
        if not assume_csv_has_header:
            kw['header'] = None
        df = pd.read_csv(p, **kw)
    df = _normalize(df)
    if 'ts' in df.columns and 'points' not in df.columns:
        df['points'] = df['ts']
    return df

def interval_overlap_score(a_start, a_end, b_start, b_end) -> float:
    a0 = pd.to_datetime(a_start, utc=True); a1 = pd.to_datetime(a_end, utc=True)
    b0 = pd.to_datetime(b_start, utc=True); b1 = pd.to_datetime(b_end, utc=True)
    if a1 < a0:
        a0, a1 = a1, a0
    if b1 < b0:
        b0, b1 = b1, b0
    inter_start = max(a0, b0); inter_end = min(a1, b1)
    inter = (inter_end - inter_start).total_seconds()
    if inter <= 0:
        return 0.0
    dur_a = (a1 - a0).total_seconds()
    dur_b = (b1 - b0).total_seconds()
    union = dur_a + dur_b - inter
    return float(inter / union) if union > 0 else 0.0

def jaccard_points(a: Iterable[Any], b: Iterable[Any]) -> float:
    def _norm(seq):
        S = set()
        if seq is None:
            return S
        for v in seq:
            if v is None:
                continue
            try:
                t = pd.to_datetime(v, utc=True)
                if pd.isna(t):
                    continue
                S.add(int(t.value))
            except Exception:
                S.add(str(v))
        return S
    A = _norm(a); B = _norm(b)
    if not A and not B:
        return 0.0
    return float(len(A & B) / len(A | B))

def load_master(source) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return load_trade_structure(source)
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    frames = []
    if p.is_file():
        df = load_trade_structure(p)
        if 'strategy' in df.columns and (df['strategy'] == '').all():
            df['strategy'] = p.stem
        frames.append(df)
    else:
        for csv in sorted(p.glob('*.csv')):
            df = load_trade_structure(csv)
            if 'strategy' in df.columns and (df['strategy'] == '').all():
                df['strategy'] = csv.stem
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=_CANON)
    out = pd.concat(frames, axis=0, ignore_index=True)
    return _normalize(out)

def prune_overlap_strategies(df: pd.DataFrame, *,
                             overlap_threshold: float = 0.75,
                             score_col: str = 'oos_mar',
                             mlflow_log: bool = False,
                             artifacts_dir = None,
                             threshold: float = None,
                             score_column: str = None,
                             corr_threshold: float = None):
    # returns dict: {'kept': [...], 'pruned': [...], 'runstamp': 'YYYYmmdd_HHMMSS'}
    if threshold is not None:
        overlap_threshold = float(threshold)
    if score_column is not None:
        score_col = str(score_column)

    kept_ids: list[str] = []
    pruned_ids: list[str] = []

    # collect ids for complement logic
    if 'strategy_id' in df.columns:
        all_ids = [str(x) for x in df['strategy_id'].tolist()]
    elif 'strategy' in df.columns:
        all_ids = [str(x) for x in df['strategy'].tolist()]
    else:
        all_ids = []

    if 'trades_csv' in df.columns and 'strategy_id' in df.columns:
        rows = []
        for _, r in df.iterrows():
            sid = str(r.get('strategy_id', r.get('strategy', '')))
            tcsv = r.get('trades_csv')
            if pd.isna(tcsv) or not tcsv:
                rows.append((sid, set(), r))
                continue
            tdf = load_trade_structure(str(tcsv))
            if 'points' in tdf.columns:
                pts = set(tdf['points'].dropna().astype(str))
            elif 'entry_time' in tdf.columns:
                pts = set(tdf['entry_time'].dropna().astype(str))
            else:
                pts = set()
            rows.append((sid, pts, r))

        def _score(rr):
            v = rr[2].get(score_col, None)
            try:
                return float(v)
            except Exception:
                return float('-inf')

        rows.sort(key=_score, reverse=True)
        kept_pts: list[set] = []
        for sid, pts, _ in rows:
            drop = False
            for p2 in kept_pts:
                if not pts and not p2:
                    continue
                if jaccard_points(pts, p2) >= float(overlap_threshold):
                    drop = True
                    break
            if drop:
                pruned_ids.append(sid)
            else:
                kept_ids.append(sid)
                kept_pts.append(pts)
    else:
        kept_ids = sorted(set(all_ids))

    if all_ids:
        pruned_ids = sorted(set(all_ids) - set(kept_ids))

    runstamp = time.strftime('%Y%m%d_%H%M%S')

    # Minimal artifacts if artifacts_dir is provided
    if artifacts_dir:
        try:
            base = os.fspath(artifacts_dir)
            folder = os.path.join(base, runstamp)
            os.makedirs(folder, exist_ok=True)
            pd.DataFrame({'kept': kept_ids, 'pruned': pruned_ids}).to_csv(os.path.join(folder, 'matrix.csv'), index=False)
            with open(os.path.join(folder, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump({'kept': kept_ids, 'pruned': pruned_ids}, f)
        except Exception:
            pass

    return {'kept': kept_ids, 'pruned': pruned_ids, 'runstamp': runstamp}

def prune_master_items(source, *,
                       threshold: float = 0.75,
                       score_column: str = 'pnl',
                       overlap_threshold: float = None,
                       score_col: str = None,
                       corr_threshold: float = None):
    # returns dict with kept/pruned and 'dropped' alias (tests expect 'dropped')
    if overlap_threshold is not None:
        threshold = float(overlap_threshold)
    if score_col is not None:
        score_column = str(score_col)

    if isinstance(source, list):
        rows = []
        for item in source:
            rid = str(item.get('run_id',''))
            metrics = item.get('metrics', {}) or {}
            arts = item.get('artifacts', {}) or {}
            rows.append({'strategy_id': rid, 'trades_csv': arts.get('trades_csv',''), 'oos_mar': metrics.get('mar', None)})
        cand = pd.DataFrame(rows)
        result = prune_overlap_strategies(cand, threshold=threshold, score_column='oos_mar', corr_threshold=corr_threshold)
        kept_ids   = result.get('kept', [])
        pruned_ids = result.get('pruned', [])
        return {'kept':   [{'run_id': k} for k in kept_ids],
                'pruned': [{'run_id': p} for p in pruned_ids],
                'dropped':[{'run_id': p} for p in pruned_ids]}
    df = load_master(source)
    return prune_overlap_strategies(df, threshold=threshold, score_column=score_column, corr_threshold=corr_threshold)

def save_master(obj, dest):
    p = Path(dest)
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, (list, dict)):
        out = p if p.suffix.lower() == '.json' else (p / 'master.json' if p.is_dir() else p.with_suffix('.json'))
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return out
    if isinstance(obj, pd.DataFrame):
        out = p if p.suffix.lower() == '.csv' else (p / 'master.csv' if p.is_dir() else p.with_suffix('.csv'))
        df = obj.copy()
        cols = list(df.columns)
        if 'timestamp' in cols:
            cols = ['timestamp'] + [c for c in cols if c != 'timestamp']
            df = df[cols]
        df.to_csv(out, index=False)
        return out
    out = p if p.suffix.lower() == '.json' else (p / 'master.json' if p.is_dir() else p.with_suffix('.json'))
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(str(obj), f, ensure_ascii=False, indent=2)
    return out
