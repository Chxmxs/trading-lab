from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from companion.ml.feature_adapter import load_latest_labels, join_features_with_labels

try:
    labels = load_latest_labels("BTCUSD","15m")
except Exception as e:
    print("[check] Could not load labels for BTCUSD@15m:", e)
    raise SystemExit(2)

# Build a tiny feature frame aligned to first 3 label timestamps
idx = labels.index[:3]
feats = pd.DataFrame({"f1":[0.1,0.2,0.3]}, index=idx)

joined = join_features_with_labels(feats, labels)
print("[check] Joined head:")
print(joined.head())
print("[check] Columns:", list(joined.columns))
print("[check] Index tz:", str(joined.index.tz))