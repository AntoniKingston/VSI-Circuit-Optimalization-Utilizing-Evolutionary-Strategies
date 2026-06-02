#!/usr/bin/env python3
import configparser
import csv
import itertools
import json
import logging
import math
import statistics
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Tuple

from algorithms import PSOMatlabConfig, optimize_pso_matlab


def _parse_list(text: str, cast=float) -> List[Any]:
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_list(text: str) -> List[int]:
    return _parse_list(text, int)


def _parse_float_list(text: str) -> List[float]:
    return _parse_list(text, float)


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _parse_pso_trajectory(log_text: str) -> List[Tuple[int, float]]:
    import re

    pattern = re.compile(r"Iteration:\s*(\d+).*Fitness\(best\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    points = []
    for line in log_text.splitlines():
        match = pattern.search(line)
        if match:
            points.append((int(match.group(1)), float(match.group(2))))
    return points


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    values_sorted = sorted(values)
    idx = (len(values_sorted) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(values_sorted[lo])
    return float(values_sorted[lo] + (values_sorted[hi] - values_sorted[lo]) * (idx - lo))


def _build_supported_pso_kwargs(seed: int, budget: int, params: Dict[str, Any]) -> Dict[str, Any]:
    supported = {f.name for f in fields(PSOMatlabConfig)}
    pso_kwargs: Dict[str, Any] = {}

    particle_number = int(params.get("particle_number", 40))
    max_iter = max(1, budget // particle_number)

    candidate = {
        "max_iter": max_iter,
        "particle_number": particle_number,
        "opt_info_interval": int(params.get("info_interval", 10)),
        "echo_logs": bool(int(params.get("echo_logs", 0))),
        "random_seed": int(seed),
        "c1": float(params.get("c1", 2.05)),
        "c2": float(params.get("c2", 2.05)),
        "rang_coef": float(params.get("rang_coef", 0.6)),
    }
    for key, value in candidate.items():
        if key in supported:
            pso_kwargs[key] = value
    return pso_kwargs


def _write_progress(progress_path: Path, total: int, done: int, ok: int, err: int, current: Dict[str, Any], best: Dict[str, Any]) -> None:
    payload = {
        "updated_at_epoch_s": time.time(),
        "total_tasks": total,
        "completed_tasks": done,
        "remaining_tasks": max(0, total - done),
        "progress_ratio": (done / total) if total else 1.0,
        "ok_count": ok,
        "error_count": err,
        "current_task": current,
        "best_so_far": best,
    }
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(config_path: str) -> None:
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")

    general = parser["general"]
    output_dir = Path(general.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / general.get("pso_debug_log_file", "pso_run_debug.txt")
    _configure_logging(log_path)

    detailed_file = output_dir / general.get("pso_detailed_results_file", "pso_detailed_results.csv")
    summary_file = output_dir / general.get("pso_summary_file", "pso_summary_results.csv")
    traj_file = output_dir / general.get("pso_trajectory_file", "pso_trajectories.csv")
    progress_file = output_dir / general.get("pso_progress_file", "pso_progress.json")

    seeds = _parse_int_list(general.get("seeds", "1,2,3,4,5"))
    budgets = _parse_int_list(general.get("budgets", "1000,2000,5000"))
    save_trajectories = general.getboolean("save_trajectories", fallback=True)

    pso_grid_section = parser["grid.pso"] if "grid.pso" in parser else {}
    particle_numbers = _parse_int_list(pso_grid_section.get("particle_number", "40"))
    info_intervals = _parse_int_list(pso_grid_section.get("info_interval", "10"))
    c1s = _parse_float_list(pso_grid_section.get("c1", "2.05"))
    c2s = _parse_float_list(pso_grid_section.get("c2", "2.05"))
    rang_coefs = _parse_float_list(pso_grid_section.get("rang_coef", "0.6"))
    echo_logs = _parse_int_list(pso_grid_section.get("echo_logs", "0"))

    grid = list(
        itertools.product(
            particle_numbers,
            info_intervals,
            c1s,
            c2s,
            rang_coefs,
            echo_logs,
        )
    )
    tasks = []
    for budget in budgets:
        for seed in seeds:
            for particle_number, info_interval, c1, c2, rang_coef, echo in grid:
                params = {
                    "particle_number": particle_number,
                    "info_interval": info_interval,
                    "c1": c1,
                    "c2": c2,
                    "rang_coef": rang_coef,
                    "echo_logs": echo,
                }
                tasks.append({"budget": budget, "seed": seed, "params": params})

    supported_fields = {f.name for f in fields(PSOMatlabConfig)}
    ignored_fields = [k for k in ["c1", "c2", "rang_coef", "random_seed"] if k not in supported_fields]
    if ignored_fields:
        logging.warning("Current PSOMatlabConfig does not support fields: %s. They will be ignored.", ignored_fields)

    detail_rows: List[Dict[str, Any]] = []
    traj_rows: List[Dict[str, Any]] = []

    total = len(tasks)
    done = 0
    ok_count = 0
    err_count = 0
    best_so_far = {"best_fitness": None}

    _write_progress(progress_file, total, done, ok_count, err_count, current={}, best=best_so_far)

    for task in tasks:
        budget = int(task["budget"])
        seed = int(task["seed"])
        params = task["params"]
        params_json = json.dumps(params, sort_keys=True)
        logging.info("Run start | budget=%s seed=%s params=%s", budget, seed, params)
        row = {
            "algorithm": "pso",
            "budget": budget,
            "seed": seed,
            "hyperparams_json": params_json,
        }
        try:
            cfg_kwargs = _build_supported_pso_kwargs(seed, budget, params)
            started = time.perf_counter()
            best_f, logs = optimize_pso_matlab(PSOMatlabConfig(**cfg_kwargs))
            runtime_s = time.perf_counter() - started
            particle_number = int(params["particle_number"])
            max_iter = max(1, budget // particle_number)
            fevals = max_iter * particle_number
            row.update(
                {
                    "status": "ok",
                    "best_fitness": float(best_f),
                    "best_solution_json": "[]",
                    "fevals": fevals,
                    "runtime_s": runtime_s,
                    "error": "",
                }
            )
            ok_count += 1
            if best_so_far["best_fitness"] is None or float(best_f) < float(best_so_far["best_fitness"]):
                best_so_far = {
                    "best_fitness": float(best_f),
                    "budget": budget,
                    "seed": seed,
                    "hyperparams_json": params_json,
                }
            if save_trajectories:
                for step, val in _parse_pso_trajectory(logs):
                    traj_rows.append(
                        {
                            "algorithm": "pso",
                            "budget": budget,
                            "seed": seed,
                            "hyperparams_json": params_json,
                            "step": step,
                            "best_so_far": val,
                        }
                    )
            logging.info("Run done  | budget=%s seed=%s best=%s", budget, seed, best_f)
        except Exception as err:
            row.update(
                {
                    "status": "error",
                    "best_fitness": math.nan,
                    "best_solution_json": "[]",
                    "fevals": 0,
                    "runtime_s": 0.0,
                    "error": str(err),
                }
            )
            err_count += 1
            logging.error("Run failed | budget=%s seed=%s params=%s | error=%s", budget, seed, params, err)

        detail_rows.append(row)
        done += 1
        _write_progress(progress_file, total, done, ok_count, err_count, current=task, best=best_so_far)

    with detailed_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "budget",
                "seed",
                "hyperparams_json",
                "status",
                "best_fitness",
                "best_solution_json",
                "fevals",
                "runtime_s",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    if save_trajectories:
        with traj_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "algorithm",
                    "budget",
                    "seed",
                    "hyperparams_json",
                    "step",
                    "best_so_far",
                ],
            )
            writer.writeheader()
            writer.writerows(traj_rows)

    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for row in detail_rows:
        key = (row["budget"], row["hyperparams_json"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (budget, params_json), rows in grouped.items():
        vals = [float(r["best_fitness"]) for r in rows if r["status"] == "ok" and not math.isnan(float(r["best_fitness"]))]
        summary_rows.append(
            {
                "algorithm": "pso",
                "budget": budget,
                "hyperparams_json": params_json,
                "n_runs": len(rows),
                "n_ok": len(vals),
                "mean": float(statistics.mean(vals)) if vals else math.nan,
                "median": float(statistics.median(vals)) if vals else math.nan,
                "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
                "min": float(min(vals)) if vals else math.nan,
                "max": float(max(vals)) if vals else math.nan,
                "q1": _quantile(vals, 0.25),
                "q3": _quantile(vals, 0.75),
            }
        )

    with summary_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "budget",
                "hyperparams_json",
                "n_runs",
                "n_ok",
                "mean",
                "median",
                "std",
                "min",
                "max",
                "q1",
                "q3",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    logging.info("PSO detailed results saved to: %s", detailed_file)
    logging.info("PSO summary results saved to: %s", summary_file)
    if save_trajectories:
        logging.info("PSO trajectories saved to: %s", traj_file)
    logging.info("PSO progress file saved to: %s", progress_file)


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Run only PSO experiments from txt config.")
    arg_parser.add_argument(
        "--config",
        default="experiments_config.txt",
        help="Path to .txt configuration file (INI format).",
    )
    args = arg_parser.parse_args()
    main(args.config)
