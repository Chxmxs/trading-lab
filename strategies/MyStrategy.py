def optuna_space(trial):
    return {
        "lookback": trial.suggest_int("lookback", 20, 200, step=5),
        "threshold": trial.suggest_float("threshold", 0.1, 2.0),
        "risk_per_trade": trial.suggest_float("risk_per_trade", 0.001, 0.02, log=True),
        "use_trailing": trial.suggest_categorical("use_trailing", [False, True]),
    }
