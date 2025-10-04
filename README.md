## New Features (Phase 8.5)

#CHATGPT PRO DID THIS XD

# 🧠 Trading-Lab – AI Strategy Explorer

This repository automates discovery, optimization, and evaluation of trading strategies using **PyBroker**, **mlfinlab**, **Optuna**, and **OpenAI-LLM**.

---

## ⚙️ Environments
| Environment | Purpose | Key Packages |
|--------------|----------|---------------|
| tradebot | Main backtesting / orchestrator | pybroker, optuna, mlflow, openai |
| tradebot-mlfl | PurgedKFold CV & meta-labeling | mlfinlab |

Activate the right one before running:
```powershell
conda activate tradebot
python -m companion.explorer.cli run --iterations 3 --data-dir .\data\usable --config .\configs\explorer.json


- Added LLM integration placeholder.
- Added basic logging setup (see `companion/logging_config.py`).
- Added self-healing framework for future patches.
- Extended CLI support for configurable cash and commission.
