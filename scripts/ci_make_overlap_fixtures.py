# -*- coding: utf-8 -*-
# scripts/ci_make_overlap_fixtures.py
from pathlib import Path
CI = Path("artifacts/_ci/overlap")
CI.mkdir(parents=True, exist_ok=True)
(CI/"trades_A.csv").write_text("entry_time\n2024-01-01T00:00:00Z\n2024-01-01T01:00:00Z\n2024-01-01T02:00:00Z\n", encoding="utf-8")
(CI/"trades_B.csv").write_text("entry_time\n2024-01-01T00:00:00Z\n2024-01-01T01:00:00Z\n2024-01-01T03:00:00Z\n", encoding="utf-8")
(CI/"trades_C.csv").write_text("entry_time\n2024-01-05T00:00:00Z\n2024-01-06T00:00:00Z\n", encoding="utf-8")
print("ci fixtures ready:", CI)
