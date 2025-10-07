# -*- coding: utf-8 -*-
"""
List what's actually inside your MLflow store (file:./mlruns).
Shows experiments, run counts, and the top metric/param/tag keys.
Real data only. No installs. Deterministic.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import mlflow

def main() -> int:
    mlflow.set_tracking_uri("file:./mlruns")
    client = mlflow.tracking.MlflowClient()

    try:
        exps = client.search_experiments()
    except Exception as e:
        print(f"[INV] Could not enumerate experiments: {e}")
        return 2

    if not exps:
        print("[INV] No experiments found in mlruns/")
        return 0

    print(f"[INV] Experiments: {len(exps)}")
    for exp in exps:
        try:
            runs = client.search_runs([exp.experiment_id], max_results=5000)
        except Exception as e:
            print(f"[INV]  - {exp.name}: error reading runs: {e}")
            continue

        m_keys, p_keys, t_keys = Counter(), Counter(), Counter()
        with_params = 0
        with_objective = 0

        for r in runs:
            md = {k.lower(): v for k, v in r.data.metrics.items()}
            pd = {k.lower(): v for k, v in r.data.params.items()}
            td = {k.lower(): v for k, v in r.data.tags.items()}

            m_keys.update(md.keys())
            p_keys.update(pd.keys())
            t_keys.update(td.keys())

            if any(k.startswith("param_") for k in pd):
                with_params += 1
            if any(k in md for k in ("oos_mar","objective","mar","expected_return_per_trade","erpt","oos_sortino","oos_sharpe")):
                with_objective += 1

        print(f"[INV]  - {exp.name}: runs={len(runs)}, with_param_*={with_params}, with_objective={with_objective}")
        def top_n(cnt: Counter, n=12):
            return ", ".join(f"{k}({v})" for k, v in cnt.most_common(n))
        print(f"[INV]    top metrics: {top_n(m_keys)}")
        print(f"[INV]    top params : {top_n(p_keys)}")
        print(f"[INV]    top tags   : {top_n(t_keys)}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
