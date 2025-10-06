from __future__ import annotations
from typing import List, Tuple, Dict, Any

def classify_error(text: str) -> str:
    if text is None: return "unknown"
    if text == "":   return "performance"
    t = str(text).lower()
    if "keyerror" in t or "missing column" in t: return "data"
    if "valueerror" in t or "shape" in t or "schema" in t: return "schema"
    if "importerror" in t or "no module named" in t: return "import"
    if "unicode" in t or "utf-8" in t or "utf-16" in t: return "encoding"
    if "json" in t and ("bom" in t or "utf" in t): return "data_bom"
    if "filenotfound" in t or "no such file" in t or "cannot find path" in t: return "missing_file"
    if "mlflow" in t and ("uri" in t or "tracking" in t): return "mlflow_uri"
    if "min trades" in t or "min_trades" in t: return "min_trades"
    if "connectionerror" in t or "timeout" in t or "temporarily unavailable" in t: return "transient"
    if "permission" in t or "access is denied" in t: return "permission"
    return "unknown"

def is_transient(label: str) -> bool:
    return label in {"transient"}

def scan_and_classify(*, experiment_name: str, outdir, client) -> Dict[str, Any]:
    import os, json
    outdir = str(outdir)
    try: os.makedirs(outdir, exist_ok=True)
    except Exception: pass
    exp = client.get_experiment_by_name(experiment_name)
    exp_id = getattr(exp, "experiment_id", "0")
    try:
        runs = client.search_runs([exp_id], "")
    except TypeError:
        runs = client.search_runs([exp_id])
    items: List[Tuple[str,str]] = []
    for r in runs:
        rid = getattr(getattr(r, "info", None), "run_id", "unknown")
        tags = getattr(getattr(r, "data", None), "tags", {}) or {}
        txt = tags.get("error_message","")
        label = classify_error(txt)
        items.append((rid, label))
        try: client.set_tag(rid, "classified_label", label)
        except Exception: pass
    # decisions.jsonl
    try:
        with open(os.path.join(outdir, "decisions.jsonl"), "w", encoding="utf-8") as f:
            for rid, lab in items:
                f.write(json.dumps({"run_id": rid, "label": lab}) + "\n")
    except Exception:
        pass
    # summary.json
    try:
        with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"count_classified": len(items)}, f)
    except Exception:
        pass
    return {"count_classified": len(items), "items": items}
