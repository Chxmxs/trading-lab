from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any, Dict
import optuna

from tuner.optuna_tuner import build_study, OptunaTuner


class PlateauEarlyStopper:
    def __init__(self, patience: int = 10, epsilon: float = 0.002):
        self.patience = int(patience)
        self.epsilon = float(epsilon)
        self.best = None
        self.no_improve = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        val = trial.value
        if val is None or val == float("-inf"):
            self.no_improve += 1
            return
        if self.best is None or (val > self.best + self.epsilon):
            self.best = val
            self.no_improve = 0
        else:
            self.no_improve += 1
        if self.no_improve >= self.patience:
            print(f"[EARLY STOP] Plateau detected: patience={self.patience}, epsilon={self.epsilon}. Stopping study.")
            study.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/tuning.json")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8-sig") as f:
        cfg: Dict[str, Any] = json.load(f)

    # --- Optional MLflow init (Windows-safe). If anything fails, we continue without MLflow. ---
    try:
        import mlflow  # optional
        mlf_cfg = cfg.get("mlflow", {}) or {}
        if mlf_cfg.get("enabled", True):
            # Use a proper file:// URI on Windows
            tracking_uri = mlf_cfg.get("tracking_uri") or (Path.cwd() / "mlruns").resolve().as_uri()
            mlflow.set_tracking_uri(tracking_uri)
            exp_name = mlf_cfg.get("experiment_name") or (cfg.get("optuna", {}).get("study_name") or "Default")
            mlflow.set_experiment(exp_name)
            print(f"[MLFLOW] tracking_uri={tracking_uri} experiment=\"{exp_name}\"")
    except Exception as e:
        print(f"[MLFLOW][WARN] init failed: {e}. Continuing without MLflow.")
    # -------------------------------------------------------------------------------------------

    study, tc = build_study(cfg)
    tuner = OptunaTuner(tc, study)

    n_trials = int(tc.optuna.get("n_trials", 50))
    plateau_patience = int(tc.optuna.get("plateau_patience", 10))
    plateau_epsilon  = float(tc.optuna.get("plateau_epsilon", 0.002))
    stopper = PlateauEarlyStopper(plateau_patience, plateau_epsilon)

    print(f"[TUNE] study={study.study_name} objective={tc.objective} strategy={tc.strategy} {tc.symbol}@{tc.timeframe}")
    study.optimize(tuner, n_trials=n_trials, callbacks=[stopper])

    best = study.best_trial if len(study.trials) else None
    if best:
        print(f"[BEST] trial={best.number} value={best.value} params={best.params}")
    else:
        print("[BEST] No completed trials.")


if __name__ == "__main__":
    main()
