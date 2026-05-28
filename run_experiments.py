#!/usr/bin/env python3
import configparser
import csv
import itertools
import json
import logging
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import algorithms
from algorithms import (
    CMAESConfig,
    ESConfig,
    JDEConfig,
    PSOMatlabConfig,
    optimize,
    optimize_cma_es,
    optimize_jde,
    optimize_pso_matlab,
)


def _parse_list(text: str, cast=float) -> List[Any]:
    items = [x.strip() for x in text.split(",") if x.strip()]
    return [cast(x) for x in items]


def _parse_int_list(text: str) -> List[int]:
    return _parse_list(text, int)


def _parse_float_list(text: str) -> List[float]:
    return _parse_list(text, float)


def _parse_str_list(text: str) -> List[str]:
    return _parse_list(text, str)


def _grid_product(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def _configure_logging(debug_log_path: Path) -> None:
    debug_log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(debug_log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _safe_mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else math.nan


def _safe_std(values: List[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    arr = np.asarray(values, dtype=float)
    return float(np.quantile(arr, q))


class FitnessProbe:
    def __init__(self, base_fitness_fn):
        self.base_fitness_fn = base_fitness_fn
        self.fevals = 0
        self.best_so_far = math.inf
        self.trajectory: List[Tuple[int, float]] = []

    def __call__(self, individual, opt_domain_pos_len=4, client_script_name="client.py"):
        value = self.base_fitness_fn(individual, opt_domain_pos_len, client_script_name)
        self.fevals += 1
        if value < self.best_so_far:
            self.best_so_far = value
        self.trajectory.append((self.fevals, self.best_so_far))
        return value


def _parse_pso_trajectory(log_text: str) -> List[Tuple[int, float]]:
    import re

    pattern = re.compile(r"Iteration:\s*(\d+).*Fitness\(best\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    points = []
    for line in log_text.splitlines():
        match = pattern.search(line)
        if match:
            points.append((int(match.group(1)), float(match.group(2))))
    return points


def _run_es(seed: int, budget: int, cfg: Dict[str, Any], client_script_name: str):
    np.random.seed(seed)
    random.seed(seed)
    lamb = int(cfg["lamb"])
    n_iterations = max(1, budget // lamb)
    es_cfg = ESConfig(
        n_iterations=n_iterations,
        mu=int(cfg["mu"]),
        lamb=lamb,
        sigma_0=float(cfg["sigma_0"]),
        sigma_sigma=float(cfg["sigma_sigma"]),
        client_script_name=client_script_name,
    )
    probe = FitnessProbe(algorithms.compute_fitness)
    old = algorithms.compute_fitness
    algorithms.compute_fitness = probe
    try:
        start = time.perf_counter()
        best_f, best_ind = optimize(es_cfg)
        runtime = time.perf_counter() - start
        return best_f, best_ind, probe.fevals, runtime, probe.trajectory
    finally:
        algorithms.compute_fitness = old


def _run_jde(seed: int, budget: int, cfg: Dict[str, Any], client_script_name: str):
    np.random.seed(seed)
    random.seed(seed)
    pop_size = int(cfg["population_size"])
    n_iterations = max(1, budget // pop_size)
    jde_cfg = JDEConfig(
        n_iterations=n_iterations,
        population_size=pop_size,
        F_init=float(cfg["F_init"]),
        CR_init=float(cfg["CR_init"]),
        tau_F=float(cfg["tau"]),
        tau_CR=float(cfg["tau"]),
        client_script_name=client_script_name,
    )
    probe = FitnessProbe(algorithms.compute_fitness)
    old = algorithms.compute_fitness
    algorithms.compute_fitness = probe
    try:
        start = time.perf_counter()
        best_f, best_ind = optimize_jde(jde_cfg)
        runtime = time.perf_counter() - start
        return best_f, best_ind, probe.fevals, runtime, probe.trajectory
    finally:
        algorithms.compute_fitness = old


def _run_cma(seed: int, budget: int, cfg: Dict[str, Any], client_script_name: str):
    np.random.seed(seed)
    random.seed(seed)
    cma_cfg = CMAESConfig(
        sigma_0=float(cfg["sigma_0"]),
        max_fevals=int(budget),
        client_script_name=client_script_name,
        seed=seed,
        cma_options={"popsize": int(cfg["popsize"])},
    )
    probe = FitnessProbe(algorithms.compute_fitness)
    old = algorithms.compute_fitness
    algorithms.compute_fitness = probe
    try:
        start = time.perf_counter()
        best_f, best_ind = optimize_cma_es(cma_cfg)
        runtime = time.perf_counter() - start
        return best_f, best_ind, probe.fevals, runtime, probe.trajectory
    finally:
        algorithms.compute_fitness = old


def _run_pso(seed: int, budget: int, cfg: Dict[str, Any], client_script_name: str):
    _ = client_script_name  # PSO runs inside MATLAB evaluator path.
    particle_number = int(cfg["particle_number"])
    max_iter = max(1, budget // particle_number)
    pso_cfg = PSOMatlabConfig(
        max_iter=max_iter,
        particle_number=particle_number,
        opt_info_interval=int(cfg["info_interval"]),
        c1=float(cfg["c1"]),
        c2=float(cfg["c2"]),
        rang_coef=float(cfg["rang_coef"]),
        random_seed=seed,
        echo_logs=bool(int(cfg["echo_logs"])),
    )
    start = time.perf_counter()
    best_f, logs = optimize_pso_matlab(pso_cfg)
    runtime = time.perf_counter() - start
    trajectory = _parse_pso_trajectory(logs)
    fevals = max_iter * particle_number
    best_ind = []
    return best_f, best_ind, fevals, runtime, trajectory


RUNNERS = {
    "es": _run_es,
    "jde": _run_jde,
    "cma": _run_cma,
    "pso": _run_pso,
}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _write_progress_file(
    progress_path: Path,
    run_label: str,
    total_tasks: int,
    completed_tasks: int,
    ok_count: int,
    error_count: int,
    detail_rows: List[Dict[str, Any]],
    current_task: Dict[str, Any],
    current_result: Dict[str, Any],
) -> None:
    best_ok = [r for r in detail_rows if r.get("status") == "ok" and not math.isnan(float(r.get("best_fitness", math.nan)))]
    best_row = min(best_ok, key=lambda r: float(r["best_fitness"])) if best_ok else None
    payload = {
        "run_label": run_label,
        "updated_at_epoch_s": time.time(),
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "remaining_tasks": max(0, total_tasks - completed_tasks),
        "progress_ratio": (completed_tasks / total_tasks) if total_tasks else 1.0,
        "ok_count": ok_count,
        "error_count": error_count,
        "current_task": _to_jsonable(current_task),
        "current_result": _to_jsonable(
            {
                "status": current_result.get("status"),
                "best_fitness": current_result.get("best_fitness"),
                "fevals": current_result.get("fevals"),
                "runtime_s": current_result.get("runtime_s"),
                "error": current_result.get("error"),
            }
        ),
        "best_so_far": _to_jsonable(
            {
                "algorithm": best_row["algorithm"],
                "budget": best_row["budget"],
                "seed": best_row["seed"],
                "hyperparams_json": best_row["hyperparams_json"],
                "best_fitness": best_row["best_fitness"],
                "runtime_s": best_row["runtime_s"],
                "fevals": best_row["fevals"],
            }
            if best_row
            else {}
        ),
    }
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _execute_single_task(task: Dict[str, Any]) -> Dict[str, Any]:
    algo_name = task["algorithm"]
    seed = int(task["seed"])
    budget = int(task["budget"])
    hyperparams = task["hyperparams"]
    client_script_name = task["client_script_name"]

    row: Dict[str, Any] = {
        "algorithm": algo_name,
        "budget": budget,
        "seed": seed,
        "hyperparams_json": json.dumps(hyperparams, sort_keys=True),
    }
    try:
        best_f, best_ind, fevals, runtime_s, trajectory = RUNNERS[algo_name](
            seed, budget, hyperparams, client_script_name
        )
        row.update(
            {
                "status": "ok",
                "best_fitness": best_f,
                "best_solution_json": json.dumps(best_ind),
                "fevals": fevals,
                "runtime_s": runtime_s,
                "error": "",
                "trajectory": trajectory,
            }
        )
    except Exception as err:
        row.update(
            {
                "status": "error",
                "best_fitness": math.nan,
                "best_solution_json": "[]",
                "fevals": 0,
                "runtime_s": 0.0,
                "error": str(err),
                "trajectory": [],
            }
        )
    return row


def main(config_path: str) -> None:
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")

    general = parser["general"]
    output_dir = Path(general.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_log_path = output_dir / general.get("debug_log_file", "run_debug.txt")
    _configure_logging(debug_log_path)

    seeds = _parse_int_list(general.get("seeds", "1,2,3"))
    budgets = _parse_int_list(general.get("budgets", "2000,5000"))
    algorithms_enabled = _parse_str_list(general.get("algorithms", "es,jde,cma,pso"))
    stats_to_collect = set(_parse_str_list(general.get("stats", "mean,median,std,min,max,q1,q3")))
    client_script_name = general.get("client_script_name", "client.py")
    save_trajectories = general.getboolean("save_trajectories", fallback=True)
    max_workers = general.getint("max_workers", fallback=max(1, os.cpu_count() or 1))
    parallel_pso = general.getboolean("parallel_pso", fallback=False)
    progress_file_name = general.get("progress_file", "progress.json")
    run_label = general.get("run_label", f"run-{int(time.time())}")
    progress_path = output_dir / progress_file_name

    logging.info("Loaded config from %s", config_path)
    logging.info(
        "Algorithms: %s | Budgets: %s | Seeds: %s | max_workers=%s | parallel_pso=%s",
        algorithms_enabled,
        budgets,
        seeds,
        max_workers,
        parallel_pso,
    )

    grid_cfg: Dict[str, List[Dict[str, Any]]] = {}
    if "grid.es" in parser:
        grid_cfg["es"] = _grid_product(
            {
                "mu": _parse_int_list(parser["grid.es"].get("mu", "20")),
                "lamb": _parse_int_list(parser["grid.es"].get("lamb", "100")),
                "sigma_0": _parse_float_list(parser["grid.es"].get("sigma_0", "0.3")),
                "sigma_sigma": _parse_float_list(parser["grid.es"].get("sigma_sigma", "0.1")),
            }
        )
    if "grid.jde" in parser:
        grid_cfg["jde"] = _grid_product(
            {
                "population_size": _parse_int_list(parser["grid.jde"].get("population_size", "40")),
                "F_init": _parse_float_list(parser["grid.jde"].get("F_init", "0.5")),
                "CR_init": _parse_float_list(parser["grid.jde"].get("CR_init", "0.9")),
                "tau": _parse_float_list(parser["grid.jde"].get("tau", "0.1")),
            }
        )
    if "grid.cma" in parser:
        grid_cfg["cma"] = _grid_product(
            {
                "sigma_0": _parse_float_list(parser["grid.cma"].get("sigma_0", "0.3")),
                "popsize": _parse_int_list(parser["grid.cma"].get("popsize", "8")),
            }
        )
    if "grid.pso" in parser:
        grid_cfg["pso"] = _grid_product(
            {
                "particle_number": _parse_int_list(parser["grid.pso"].get("particle_number", "40")),
                "info_interval": _parse_int_list(parser["grid.pso"].get("info_interval", "10")),
                "c1": _parse_float_list(parser["grid.pso"].get("c1", "2.05")),
                "c2": _parse_float_list(parser["grid.pso"].get("c2", "2.05")),
                "rang_coef": _parse_float_list(parser["grid.pso"].get("rang_coef", "0.6")),
                "echo_logs": _parse_int_list(parser["grid.pso"].get("echo_logs", "0")),
            }
        )

    detail_rows: List[Dict[str, Any]] = []
    traj_rows: List[Dict[str, Any]] = []
    tasks_parallel: List[Dict[str, Any]] = []
    tasks_serial: List[Dict[str, Any]] = []

    for algo_name in algorithms_enabled:
        if algo_name not in RUNNERS:
            logging.warning("Skipping unknown algorithm '%s'", algo_name)
            continue
        algo_grid = grid_cfg.get(algo_name, [{}])
        for budget in budgets:
            for hyperparams in algo_grid:
                for seed in seeds:
                    task = {
                        "algorithm": algo_name,
                        "budget": budget,
                        "seed": seed,
                        "hyperparams": hyperparams,
                        "client_script_name": client_script_name,
                    }
                    if algo_name == "pso" and not parallel_pso:
                        tasks_serial.append(task)
                    else:
                        tasks_parallel.append(task)

    total_tasks = len(tasks_parallel) + len(tasks_serial)
    completed_tasks = 0
    ok_count = 0
    error_count = 0
    logging.info("Scheduled %d parallel tasks and %d serial tasks (total=%d).", len(tasks_parallel), len(tasks_serial), total_tasks)
    _write_progress_file(
        progress_path,
        run_label,
        total_tasks,
        completed_tasks,
        ok_count,
        error_count,
        detail_rows,
        current_task={},
        current_result={"status": "init"},
    )

    if tasks_parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_execute_single_task, task): task for task in tasks_parallel}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result_row = future.result()
                except Exception as err:
                    result_row = {
                        "algorithm": task["algorithm"],
                        "budget": task["budget"],
                        "seed": task["seed"],
                        "hyperparams_json": json.dumps(task["hyperparams"], sort_keys=True),
                        "status": "error",
                        "best_fitness": math.nan,
                        "best_solution_json": "[]",
                        "fevals": 0,
                        "runtime_s": 0.0,
                        "error": f"Worker failure: {err}",
                        "trajectory": [],
                    }
                detail_rows.append({k: v for k, v in result_row.items() if k != "trajectory"})
                completed_tasks += 1
                if result_row["status"] == "ok":
                    ok_count += 1
                else:
                    error_count += 1
                if save_trajectories:
                    for step, val in result_row.get("trajectory", []):
                        traj_rows.append(
                            {
                                "algorithm": result_row["algorithm"],
                                "budget": result_row["budget"],
                                "seed": result_row["seed"],
                                "hyperparams_json": result_row["hyperparams_json"],
                                "step": step,
                                "best_so_far": val,
                            }
                        )
                if result_row["status"] == "ok":
                    logging.info(
                        "Run done  | algo=%s budget=%s seed=%s best=%.6g fevals=%s runtime=%.3fs",
                        result_row["algorithm"],
                        result_row["budget"],
                        result_row["seed"],
                        result_row["best_fitness"],
                        result_row["fevals"],
                        result_row["runtime_s"],
                    )
                else:
                    logging.error(
                        "Run failed | algo=%s budget=%s seed=%s params=%s | error=%s",
                        task["algorithm"],
                        task["budget"],
                        task["seed"],
                        task["hyperparams"],
                        result_row["error"],
                    )
                _write_progress_file(
                    progress_path,
                    run_label,
                    total_tasks,
                    completed_tasks,
                    ok_count,
                    error_count,
                    detail_rows,
                    current_task=task,
                    current_result=result_row,
                )

    for task in tasks_serial:
        logging.info(
            "Run start (serial) | algo=%s budget=%s seed=%s params=%s",
            task["algorithm"],
            task["budget"],
            task["seed"],
            task["hyperparams"],
        )
        result_row = _execute_single_task(task)
        detail_rows.append({k: v for k, v in result_row.items() if k != "trajectory"})
        completed_tasks += 1
        if result_row["status"] == "ok":
            ok_count += 1
        else:
            error_count += 1
        if save_trajectories:
            for step, val in result_row.get("trajectory", []):
                traj_rows.append(
                    {
                        "algorithm": result_row["algorithm"],
                        "budget": result_row["budget"],
                        "seed": result_row["seed"],
                        "hyperparams_json": result_row["hyperparams_json"],
                        "step": step,
                        "best_so_far": val,
                    }
                )
        if result_row["status"] == "ok":
            logging.info(
                "Run done  | algo=%s budget=%s seed=%s best=%.6g fevals=%s runtime=%.3fs",
                result_row["algorithm"],
                result_row["budget"],
                result_row["seed"],
                result_row["best_fitness"],
                result_row["fevals"],
                result_row["runtime_s"],
            )
        else:
            logging.error(
                "Run failed | algo=%s budget=%s seed=%s params=%s | error=%s",
                task["algorithm"],
                task["budget"],
                task["seed"],
                task["hyperparams"],
                result_row["error"],
            )
        _write_progress_file(
            progress_path,
            run_label,
            total_tasks,
            completed_tasks,
            ok_count,
            error_count,
            detail_rows,
            current_task=task,
            current_result=result_row,
        )

    detailed_csv = output_dir / general.get("detailed_results_file", "detailed_results.csv")
    with detailed_csv.open("w", newline="", encoding="utf-8") as f:
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
        trajectory_csv = output_dir / general.get("trajectory_file", "trajectories.csv")
        with trajectory_csv.open("w", newline="", encoding="utf-8") as f:
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

    summary_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for row in detail_rows:
        key = (row["algorithm"], row["budget"], row["hyperparams_json"])
        grouped.setdefault(key, []).append(row)

    for (algo, budget, params_json), rows in grouped.items():
        ok_vals = [float(r["best_fitness"]) for r in rows if r["status"] == "ok" and not math.isnan(float(r["best_fitness"]))]
        summary = {
            "algorithm": algo,
            "budget": budget,
            "hyperparams_json": params_json,
            "n_runs": len(rows),
            "n_ok": len(ok_vals),
        }
        if "mean" in stats_to_collect:
            summary["mean"] = _safe_mean(ok_vals)
        if "median" in stats_to_collect:
            summary["median"] = float(statistics.median(ok_vals)) if ok_vals else math.nan
        if "std" in stats_to_collect:
            summary["std"] = _safe_std(ok_vals)
        if "min" in stats_to_collect:
            summary["min"] = float(min(ok_vals)) if ok_vals else math.nan
        if "max" in stats_to_collect:
            summary["max"] = float(max(ok_vals)) if ok_vals else math.nan
        if "q1" in stats_to_collect:
            summary["q1"] = _quantile(ok_vals, 0.25)
        if "q3" in stats_to_collect:
            summary["q3"] = _quantile(ok_vals, 0.75)
        summary_rows.append(summary)

    summary_csv = output_dir / general.get("summary_file", "summary_results.csv")
    summary_columns = ["algorithm", "budget", "hyperparams_json", "n_runs", "n_ok", *sorted(stats_to_collect)]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summary_rows)

    logging.info("Detailed results saved to: %s", detailed_csv)
    logging.info("Summary results saved to: %s", summary_csv)
    if save_trajectories:
        logging.info("Trajectories saved to: %s", output_dir / general.get("trajectory_file", "trajectories.csv"))
    logging.info("Progress file saved to: %s", progress_path)


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Run VSI optimization experiments from txt config.")
    arg_parser.add_argument(
        "--config",
        default="experiments_config.txt",
        help="Path to .txt configuration file (INI format).",
    )
    args = arg_parser.parse_args()
    main(args.config)
