"""
Command‑line utility to scaffold a new trading strategy file.

This script generates a new Python module in the ``strategies``
directory with boilerplate code for the ``run`` method and an
``optuna_space`` description.  It also includes adapter stubs to
ensure that the strategy outputs meet the companion's expectations
(UTC equity index, trade schema).  The goal is to make it easy to
add new strategies without duplicating boilerplate.

Run this script as follows:

    python scripts/make_strategy.py --name MyNewStrategy

The script will create ``strategies/MyNewStrategy.py`` with a class
``MyNewStrategy`` and default settings.  It will not overwrite
existing files unless ``--force`` is provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path

TEMPLATE = """\"\"\"\"
import pandas as pd
from typing import Any, Dict

class {class_name}:
    \"\"\"Generated strategy template.

    This strategy demonstrates the required ``run`` method and the
    optional ``optuna_space`` static method.  Adjust the logic to
    implement your trading idea.  The ``run`` method must return a
    dictionary with keys ``equity`` (a pandas.Series indexed by UTC
    DatetimeIndex) and ``trades`` (a pandas.DataFrame with a
    ``pnl_pct`` column and timestamp columns).
    \"\"\"

    def __init__(self) -> None:
        pass

    @staticmethod
    def optuna_space() -> Dict[str, Any]:
        \"\"\"Describe the hyperparameter search space for this strategy.\"\"\"
        return {
            # Add hyperparameters here, e.g.:
            # "lookback": {"type": "int", "low": 5, "high": 30},
        }

    def run(self, df: pd.DataFrame | None, run_config: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Execute the strategy on the given DataFrame or self‑loaded data.\"\"\"
        # Load your data if df is None.  For example:
        # if df is None:
        #     df = pd.read_csv(run_config.get("price_csv", "prices.csv"), parse_dates=[0], index_col=0)
        # Compute signals and trades here.
        # Build equity series and trades DataFrame.
        idx = pd.date_range(start="2025-01-01", periods=10, freq="D", tz="UTC")
        equity = pd.Series([1.0] * len(idx), index=idx)
        trades = pd.DataFrame({
            "exit_time": [idx[0]],
            "pnl_pct": [0.0],
        })
        return {"equity": equity, "trades": trades}
\"\"\"\"

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a new strategy template.")
    parser.add_argument("--name", required=True, help="Name of the new strategy class")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()
    class_name = args.name
    file_name = f"{class_name}.py"
    strategies_dir = Path("strategies")
    strategies_dir.mkdir(exist_ok=True)
    target = strategies_dir / file_name
    if target.exists() and not args.force:
        raise FileExistsError(f"{target} already exists. Use --force to overwrite.")
    content = TEMPLATE.format(class_name=class_name)
    target.write_text(content, encoding="utf-8")
    print(f"Created {target}")

if __name__ == "__main__":  # pragma: no cover
    main()
"""
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a new strategy template.")
    parser.add_argument("--name", required=True, help="Name of the new strategy class")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()
    class_name = args.name
    file_name = f"{class_name}.py"
    strategies_dir = Path("strategies")
    strategies_dir.mkdir(exist_ok=True)
    target = strategies_dir / file_name
    if target.exists() and not args.force:
        raise FileExistsError(f"{target} already exists. Use --force to overwrite.")
    content = TEMPLATE.format(class_name=class_name)
    target.write_text(content, encoding="utf-8")
    print(f"Created {target}")

if __name__ == "__main__":  # pragma: no cover
    main()
