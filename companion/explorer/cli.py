import argparse
import json
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from companion.explorer.plot_utils import plot_equity
import mlflow

from companion.logging_config import setup_logging
from companion.explorer.discovery import (
    StrategyCandidate,
    generate_parameter_sweep,
    mutate_strategy,
    propose_new_strategy_via_llm,
    register_strategy_from_script,
)
from companion.explorer.evaluation import CVConfig, evaluate_strategy_cv
from companion.explorer.ranking import StrategyResult, update_master_list

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def discover_strategies(config: dict) -> list:
    candidates = []
    # Manual strategies
    for script_path in config.get("manual_strategies", []):
        candidate = register_strategy_from_script(script_path)
        if candidate:
            candidates.append(candidate)
    # Base strategies
    for entry in config.get("base_strategies", []):
        module_name = entry["module"]
        func_name = entry["function"]
        search_space = entry.get("search_space", {})
        mutations = entry.get("mutations", {})
        max_candidates = entry.get("max_candidates", 5)
        mod = __import__(module_name, fromlist=[func_name])
        base_func = getattr(mod, func_name)
        if search_space:
            candidates.extend(generate_parameter_sweep(base_func, search_space, max_candidates))
        else:
            candidates.append(StrategyCandidate(name=func_name, func=base_func, params={}, source="manual"))
        if mutations:
            perturb = {k: tuple(v) for k, v in mutations.items()}
            mutated = mutate_strategy(candidates[-1], perturb)
            candidates.append(mutated)
    # LLM proposals (optional)
    if config.get("llm_enabled"):
        llm_client = None
        prompt = config.get("llm_prompt", "")
        tools = config.get("llm_tools")
        llm_candidate = propose_new_strategy_via_llm(llm_client, prompt, tools)
        if llm_candidate:
            candidates.append(llm_candidate)
    return candidates

def load_data_sources(data_dirs: list) -> list:
    data_frames = []
    for d in data_dirs:
        path = Path(d)
        if not path.is_dir():
            logging.warning(f"Data directory {d} not found; skipping.")
            continue
        for file_path in path.rglob("*"):
            if file_path.suffix.lower() in (".csv", ".parquet"):
                try:
                    # Load the DataFrame
                    if file_path.suffix.lower() == ".csv":
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_parquet(file_path)
                    if len(df) == 0:
                        continue
                    # Check for a timestamp or date column
                    lower_cols = {col.lower() for col in df.columns}
                    if not ("timestamp" in lower_cols or "date" in lower_cols):
                        logging.warning(f"Skipping {file_path} due to missing timestamp/date column.")
                        continue
                    # Only append if the DataFrame has a valid time column
                    data_frames.append(df)
                except Exception as exc:
                    logging.exception(f"Failed to load {file_path}: {exc}")
    return data_frames


def run_explorer(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    cv_conf_dict = config.get("cross_validation", {})
    cv_config = CVConfig(
        n_splits=cv_conf_dict.get("n_splits", 2),
        embargo_pct=cv_conf_dict.get("embargo_pct", 0.0),
        purge_length=cv_conf_dict.get("purge_length"),
    )
    data_frames = load_data_sources(args.data_dir)
    if not data_frames:
        logging.error("No data files were loaded. Check the data directories.")
        return
    candidates = discover_strategies(config)
    if not candidates:
        logging.error("No strategies discovered. Review your configuration.")
        return
    results = []
    equities = {}

    for i, candidate in enumerate(tqdm(candidates[: args.iterations], desc="Evaluating strategies", unit="strategy")):
        logging.info(f"Evaluating strategy {candidate.name} ({i+1}/{args.iterations})...")
        agg_metrics = {"cagr": 0.0, "max_drawdown": 0.0, "mar": 0.0, "sharpe": 0.0, "num_trades": 0.0}
        count = 0
        for df in data_frames:
            metrics = evaluate_strategy_cv(
                candidate.func,
                df,
                run_config=config.get("run_config", {}),
                cv_config=cv_config
            )
            if not metrics:
                continue
            for k, v in metrics.items():
                agg_metrics[k] += v
            count += 1
        if (i + 1) % 5 == 0:
            response = input("Press Enter to continue or type 'quit' to stop: ")
            if response.strip().lower() == "quit":
                break

        # Find a data frame with OHLCV columns to compute an equity curve
        equity_series = None
        required_cols = {"open", "high", "low", "close"}
        for df in data_frames:
            # Normalize column names to lowercase for comparison
            lower_cols = {c.lower() for c in df.columns}
            if required_cols.issubset(lower_cols):
                try:
                    res = candidate.func(df, config.get("run_config", {}))
                    equity_series = res["equity"]
                    break
                except Exception:
                    continue
        equities[candidate.name] = equity_series.tolist() if equity_series is not None else []

        if equity_series is not None:
            plot_equity(
                equity_series,
                title=f"Equity Curve for {candidate.name}",
                filename=f"{candidate.name}_equity.png"
            )

        if count > 0:
            for k in agg_metrics:
                agg_metrics[k] /= count

        mlflow.start_run(run_name=candidate.name)
        mlflow.log_metrics(agg_metrics)
        mlflow.log_param("source", candidate.source)
        mlflow.end_run()

        result = StrategyResult(name=candidate.name, metrics=agg_metrics, source=candidate.source)
        results.append(result)

    updated = update_master_list(
        args.master_list,
        results,
        correlation_matrix=None,
        correlation_threshold=config.get("correlation_threshold", 0.95)
    )

    # Save equities for correlation analysis
    with open("equities.json", "w", encoding="utf-8") as eq_file:
        json.dump(equities, eq_file, ensure_ascii=False)

    print("\\nUpdated Master List (top 10):")
    for res in updated[:10]:
        print(f"{res.rank:>3}. {res.name:<30} Score: {res.score:.3f} Source: {res.source}")

def main(*argv):
    setup_logging()
    parser = argparse.ArgumentParser(description="Strategy Explorer CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")
    run_parser = subparsers.add_parser("run", help="Generate, evaluate and rank strategies")
    run_parser.add_argument("--iterations", type=int, default=10, help="Number of strategy candidates to evaluate.")
    run_parser.add_argument("--data-dir", nargs="+", default=[], help="Data directories (CSV or parquet).")
    run_parser.add_argument("--config", type=str, required=True, help="Path to explorer configuration JSON.")
    run_parser.add_argument("--master-list", type=str, default=str(Path("companion\\master_list.json")), help="Path to master list JSON.")
    run_parser.add_argument("--cash", type=float, default=100000, help="Initial cash balance")
    run_parser.add_argument("--commission", type=float, default=0.0, help="Commission per trade")
    args = parser.parse_args(list(argv) or None)
    if args.command == "run":
        run_explorer(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
