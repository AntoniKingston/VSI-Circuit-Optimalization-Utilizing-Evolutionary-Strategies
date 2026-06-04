#!/usr/bin/env python3
"""General plotting utilities for VSI optimization experiment CSV outputs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ALGO_LABELS = {
    "es": "ES",
    "jde": "jDE",
    "cma": "CMA-ES",
    "pso": "PSO",
}

ALGO_COLORS = {
    "es": "#4C72B0",
    "jde": "#55A868",
    "cma": "#C44E52",
    "pso": "#8172B3",
}


@dataclass
class PlotConfig:
    detailed_csv: Path
    output_dir: Path
    summary_csv: Optional[Path] = None
    trajectory_csv: Optional[Path] = None
    budgets: Optional[List[int]] = None
    algorithms: Optional[List[str]] = None
    fig_format: str = "pdf"
    dpi: int = 300
    figsize_single: Tuple[float, float] = (6.0, 4.0)
    figsize_wide: Tuple[float, float] = (7.2, 4.2)
    fitness_log_scale: bool = True
    plot_ids: Optional[List[str]] = None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_median(values: Sequence[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(statistics.median(vals)) if vals else math.nan


def _safe_quantile(values: Sequence[float], q: float) -> float:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return math.nan
    idx = (len(vals) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] + (vals[hi] - vals[lo]) * (idx - lo))


def _filter_rows(
    rows: List[Dict[str, str]],
    budgets: Optional[List[int]],
    algorithms: Optional[List[str]],
) -> List[Dict[str, str]]:
    filtered = rows
    if budgets is not None:
        budget_set = {str(b) for b in budgets}
        filtered = [r for r in filtered if r.get("budget", "") in budget_set]
    if algorithms is not None:
        algo_set = {a.lower() for a in algorithms}
        filtered = [r for r in filtered if r.get("algorithm", "").lower() in algo_set]
    return filtered


def _algo_label(algo: str) -> str:
    return ALGO_LABELS.get(algo.lower(), algo.upper())


def _algo_color(algo: str) -> str:
    return ALGO_COLORS.get(algo.lower(), "#333333")


def _save_figure(fig: plt.Figure, output_path: Path, dpi: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _discovered_algorithms(rows: Iterable[Dict[str, str]]) -> List[str]:
    algos = sorted({r.get("algorithm", "").lower() for r in rows if r.get("algorithm")})
    return algos


def _discovered_budgets(rows: Iterable[Dict[str, str]]) -> List[int]:
    budgets = sorted({int(r["budget"]) for r in rows if r.get("budget", "").isdigit()})
    return budgets


def plot_algorithm_fitness_boxplot(
    rows: List[Dict[str, str]],
    output_dir: Path,
    budget: int,
    fig_format: str,
    dpi: int,
    figsize: Tuple[float, float],
    log_scale: bool,
) -> Path:
    fig, ax = plt.subplots(figsize=figsize)
    algos = _discovered_algorithms(r for r in rows if r.get("budget") == str(budget))
    data = []
    labels = []
    colors = []
    for algo in algos:
        vals = [
            _to_float(r["best_fitness"])
            for r in rows
            if r.get("algorithm") == algo
            and r.get("budget") == str(budget)
            and r.get("status") == "ok"
        ]
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            continue
        data.append(vals)
        labels.append(_algo_label(algo))
        colors.append(_algo_color(algo))

    if not data:
        raise RuntimeError(f"No successful runs to plot for budget={budget}.")

    box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_title(f"Best fitness distribution (budget = {budget})")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Best fitness (lower is better)")
    if log_scale:
        positive = [v for group in data for v in group if v > 0]
        if positive and max(positive) / min(positive) > 20:
            ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)

    return _save_figure(
        fig,
        output_dir / f"algorithm_fitness_boxplot_budget_{budget}.{fig_format}",
        dpi,
    )


def plot_budget_scaling(
    rows: List[Dict[str, str]],
    output_dir: Path,
    fig_format: str,
    dpi: int,
    figsize: Tuple[float, float],
) -> Path:
    fig, ax = plt.subplots(figsize=figsize)
    algos = _discovered_algorithms(rows)
    budgets = _discovered_budgets(rows)

    for algo in algos:
        medians = []
        q1s = []
        q3s = []
        for budget in budgets:
            vals = [
                _to_float(r["best_fitness"])
                for r in rows
                if r.get("algorithm") == algo
                and r.get("budget") == str(budget)
                and r.get("status") == "ok"
            ]
            vals = [v for v in vals if not math.isnan(v)]
            medians.append(_safe_median(vals))
            q1s.append(_safe_quantile(vals, 0.25))
            q3s.append(_safe_quantile(vals, 0.75))

        color = _algo_color(algo)
        ax.plot(budgets, medians, marker="o", label=_algo_label(algo), color=color, linewidth=2)
        ax.fill_between(budgets, q1s, q3s, color=color, alpha=0.15)

    ax.set_title("Median best fitness vs evaluation budget")
    ax.set_xlabel("Function evaluation budget")
    ax.set_ylabel("Best fitness (lower is better)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    return _save_figure(fig, output_dir / f"budget_scaling_median_iqr.{fig_format}", dpi)


def plot_success_rate(
    rows: List[Dict[str, str]],
    output_dir: Path,
    fig_format: str,
    dpi: int,
    figsize: Tuple[float, float],
) -> Path:
    fig, ax = plt.subplots(figsize=figsize)
    algos = _discovered_algorithms(rows)
    ok_rates = []
    labels = []
    colors = []

    for algo in algos:
        group = [r for r in rows if r.get("algorithm") == algo]
        if not group:
            continue
        ok = sum(1 for r in group if r.get("status") == "ok")
        ok_rates.append(100.0 * ok / len(group))
        labels.append(_algo_label(algo))
        colors.append(_algo_color(algo))

    bars = ax.bar(labels, ok_rates, color=colors, alpha=0.85)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Successful runs [%]")
    ax.set_title("Evaluation success rate by algorithm")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, rate in zip(bars, ok_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 1.5, f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
    return _save_figure(fig, output_dir / f"success_rate_by_algorithm.{fig_format}", dpi)


def plot_runtime_boxplot(
    rows: List[Dict[str, str]],
    output_dir: Path,
    fig_format: str,
    dpi: int,
    figsize: Tuple[float, float],
) -> Path:
    fig, ax = plt.subplots(figsize=figsize)
    algos = _discovered_algorithms(rows)
    data = []
    labels = []
    colors = []
    for algo in algos:
        vals = [
            _to_float(r["runtime_s"])
            for r in rows
            if r.get("algorithm") == algo and r.get("status") == "ok"
        ]
        vals = [v for v in vals if not math.isnan(v) and v >= 0]
        if not vals:
            continue
        data.append(vals)
        labels.append(_algo_label(algo))
        colors.append(_algo_color(algo))

    box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_title("Runtime distribution for successful runs")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Runtime [s]")
    ax.grid(True, axis="y", alpha=0.25)
    return _save_figure(fig, output_dir / f"runtime_boxplot.{fig_format}", dpi)


def plot_ecdf_fitness(
    rows: List[Dict[str, str]],
    output_dir: Path,
    budget: int,
    fig_format: str,
    dpi: int,
    figsize: Tuple[float, float],
) -> Path:
    fig, ax = plt.subplots(figsize=figsize)
    algos = _discovered_algorithms(r for r in rows if r.get("budget") == str(budget))

    for algo in algos:
        vals = sorted(
            v
            for v in (
                _to_float(r["best_fitness"])
                for r in rows
                if r.get("algorithm") == algo
                and r.get("budget") == str(budget)
                and r.get("status") == "ok"
            )
            if not math.isnan(v)
        )
        if not vals:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", label=_algo_label(algo), color=_algo_color(algo), linewidth=2)

    ax.set_title(f"ECDF of best fitness (budget = {budget})")
    ax.set_xlabel("Best fitness (lower is better)")
    ax.set_ylabel("Fraction of runs")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    return _save_figure(fig, output_dir / f"ecdf_fitness_budget_{budget}.{fig_format}", dpi)


def plot_best_config_ranking(
    summary_rows: List[Dict[str, str]],
    output_dir: Path,
    budget: int,
    top_n: int,
    fig_format: str,
    dpi: int,
    figsize: Tuple[float, float],
) -> Path:
    candidates = [
        r
        for r in summary_rows
        if r.get("budget") == str(budget) and _to_float(r.get("median", "nan")) == _to_float(r.get("median", "nan"))
    ]
    ranked = sorted(
        candidates,
        key=lambda r: (_to_float(r.get("median", "inf")), _to_float(r.get("std", "inf"))),
    )[:top_n]

    if not ranked:
        raise RuntimeError(f"No summary rows available for budget={budget}.")

    labels = []
    values = []
    colors = []
    for row in ranked:
        algo = row.get("algorithm", "?")
        labels.append(f"{_algo_label(algo)}\n{row.get('hyperparams_json', '')[:42]}...")
        values.append(_to_float(row.get("median", "nan")))
        colors.append(_algo_color(algo))

    fig, ax = plt.subplots(figsize=(max(figsize[0], 8.0), figsize[1]))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, alpha=0.85)
    ax.set_yticks(y_pos, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Median best fitness")
    ax.set_title(f"Top {len(ranked)} configurations by median fitness (budget = {budget})")
    ax.grid(True, axis="x", alpha=0.25)
    return _save_figure(fig, output_dir / f"top_configs_budget_{budget}.{fig_format}", dpi)


def plot_convergence_curves(
    trajectory_rows: List[Dict[str, str]],
    output_dir: Path,
    budget: int,
    fig_format: str,
    dpi: int,
    figsize: Tuple[float, float],
) -> Path:
    fig, ax = plt.subplots(figsize=figsize)
    algos = _discovered_algorithms(r for r in trajectory_rows if r.get("budget") == str(budget))

    for algo in algos:
        grouped: Dict[Tuple[str, str], List[Tuple[int, float]]] = {}
        for row in trajectory_rows:
            if row.get("algorithm") != algo or row.get("budget") != str(budget):
                continue
            key = (row.get("seed", ""), row.get("hyperparams_json", ""))
            step = int(float(row.get("step", "0")))
            val = _to_float(row.get("best_so_far", "nan"))
            if math.isnan(val):
                continue
            grouped.setdefault(key, []).append((step, val))

        if not grouped:
            continue

        max_len = max(len(points) for points in grouped.values())
        matrix = np.full((len(grouped), max_len), np.nan)
        for idx, points in enumerate(grouped.values()):
            points = sorted(points, key=lambda x: x[0])
            for j, (_, val) in enumerate(points):
                matrix[idx, j] = val

        median_curve = np.nanmedian(matrix, axis=0)
        q1_curve = np.nanquantile(matrix, 0.25, axis=0)
        q3_curve = np.nanquantile(matrix, 0.75, axis=0)
        x = np.arange(1, len(median_curve) + 1)
        color = _algo_color(algo)
        ax.plot(x, median_curve, label=_algo_label(algo), color=color, linewidth=2)
        ax.fill_between(x, q1_curve, q3_curve, color=color, alpha=0.15)

    ax.set_title(f"Convergence curves (budget = {budget})")
    ax.set_xlabel("Progress index")
    ax.set_ylabel("Best-so-far fitness")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    return _save_figure(fig, output_dir / f"convergence_budget_{budget}.{fig_format}", dpi)


def plot_hyperparam_sensitivity(
    summary_rows: List[Dict[str, str]],
    output_dir: Path,
    algorithm: str,
    budget: int,
    param_name: str,
    fig_format: str,
    dpi: int,
    figsize: Tuple[float, float],
) -> Optional[Path]:
    rows = [
        r
        for r in summary_rows
        if r.get("algorithm") == algorithm and r.get("budget") == str(budget)
    ]
    if not rows:
        return None

    grouped: Dict[str, List[float]] = {}
    for row in rows:
        try:
            params = json.loads(row.get("hyperparams_json", "{}"))
        except json.JSONDecodeError:
            continue
        if param_name not in params:
            continue
        key = str(params[param_name])
        val = _to_float(row.get("median", "nan"))
        if math.isnan(val):
            continue
        grouped.setdefault(key, []).append(val)

    if len(grouped) < 2:
        return None

    labels = sorted(grouped.keys(), key=lambda x: float(x) if _is_number(x) else x)
    medians = [_safe_median(grouped[label]) for label in labels]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(len(labels)), medians, marker="o", color=_algo_color(algorithm), linewidth=2)
    ax.set_xticks(range(len(labels)), labels, rotation=0)
    ax.set_title(f"{_algo_label(algorithm)} sensitivity to {param_name} (budget = {budget})")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Median best fitness")
    ax.grid(True, alpha=0.25)
    filename = f"sensitivity_{algorithm}_{param_name}_budget_{budget}.{fig_format}"
    return _save_figure(fig, output_dir / filename, dpi)


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


PLOT_REGISTRY = {
    "algorithm_fitness_boxplot": "plot_algorithm_fitness_boxplot",
    "budget_scaling": "plot_budget_scaling",
    "success_rate": "plot_success_rate",
    "runtime_boxplot": "plot_runtime_boxplot",
    "ecdf_fitness": "plot_ecdf_fitness",
    "best_config_ranking": "plot_best_config_ranking",
    "convergence_curves": "plot_convergence_curves",
}


def generate_plots(config: PlotConfig) -> List[Path]:
    detailed_rows = _filter_rows(_read_csv(config.detailed_csv), config.budgets, config.algorithms)
    summary_rows = (
        _filter_rows(_read_csv(config.summary_csv), config.budgets, config.algorithms)
        if config.summary_csv and config.summary_csv.exists()
        else []
    )
    trajectory_rows = (
        _filter_rows(_read_csv(config.trajectory_csv), config.budgets, config.algorithms)
        if config.trajectory_csv and config.trajectory_csv.exists()
        else []
    )

    budgets = config.budgets or _discovered_budgets(detailed_rows)
    plot_ids = config.plot_ids or list(PLOT_REGISTRY.keys())
    output_paths: List[Path] = []

    for plot_id in plot_ids:
        if plot_id == "algorithm_fitness_boxplot":
            for budget in budgets:
                output_paths.append(
                    plot_algorithm_fitness_boxplot(
                        detailed_rows,
                        config.output_dir,
                        budget,
                        config.fig_format,
                        config.dpi,
                        config.figsize_single,
                        config.fitness_log_scale,
                    )
                )
        elif plot_id == "budget_scaling":
            output_paths.append(
                plot_budget_scaling(
                    detailed_rows,
                    config.output_dir,
                    config.fig_format,
                    config.dpi,
                    config.figsize_wide,
                )
            )
        elif plot_id == "success_rate":
            output_paths.append(
                plot_success_rate(
                    detailed_rows,
                    config.output_dir,
                    config.fig_format,
                    config.dpi,
                    config.figsize_single,
                )
            )
        elif plot_id == "runtime_boxplot":
            output_paths.append(
                plot_runtime_boxplot(
                    detailed_rows,
                    config.output_dir,
                    config.fig_format,
                    config.dpi,
                    config.figsize_single,
                )
            )
        elif plot_id == "ecdf_fitness":
            for budget in budgets:
                output_paths.append(
                    plot_ecdf_fitness(
                        detailed_rows,
                        config.output_dir,
                        budget,
                        config.fig_format,
                        config.dpi,
                        config.figsize_single,
                    )
                )
        elif plot_id == "best_config_ranking":
            if not summary_rows:
                continue
            for budget in budgets:
                output_paths.append(
                    plot_best_config_ranking(
                        summary_rows,
                        config.output_dir,
                        budget,
                        top_n=10,
                        fig_format=config.fig_format,
                        dpi=config.dpi,
                        figsize=config.figsize_wide,
                    )
                )
        elif plot_id == "convergence_curves":
            if not trajectory_rows:
                continue
            for budget in budgets:
                output_paths.append(
                    plot_convergence_curves(
                        trajectory_rows,
                        config.output_dir,
                        budget,
                        config.fig_format,
                        config.dpi,
                        config.figsize_wide,
                    )
                )
        elif plot_id == "hyperparam_sensitivity":
            if not summary_rows:
                continue
            sensitivity_specs = [
                ("jde", "population_size"),
                ("jde", "F_init"),
                ("es", "mu"),
                ("cma", "sigma_0"),
                ("pso", "particle_number"),
            ]
            for algo, param in sensitivity_specs:
                if config.algorithms and algo not in {a.lower() for a in config.algorithms}:
                    continue
                for budget in budgets:
                    path = plot_hyperparam_sensitivity(
                        summary_rows,
                        config.output_dir,
                        algo,
                        budget,
                        param,
                        config.fig_format,
                        config.dpi,
                        config.figsize_single,
                    )
                    if path is not None:
                        output_paths.append(path)

    manifest = config.output_dir / "plot_manifest.txt"
    manifest.write_text("\n".join(str(p) for p in output_paths), encoding="utf-8")
    output_paths.append(manifest)
    return output_paths


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Generate plots from experiment CSV results.")
    parser.add_argument("--detailed-csv", default="results/detailed_results_merged.csv")
    parser.add_argument("--summary-csv", default="results/summary_results_merged.csv")
    parser.add_argument("--trajectory-csv", default="results/trajectories.csv")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--budgets", default="", help="Comma-separated budgets, e.g. 1000,2000,5000")
    parser.add_argument("--algorithms", default="", help="Comma-separated algorithms, e.g. es,jde,cma,pso")
    parser.add_argument("--fig-format", default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--plots",
        default="algorithm_fitness_boxplot,budget_scaling,success_rate,runtime_boxplot,ecdf_fitness,best_config_ranking,hyperparam_sensitivity",
        help="Comma-separated plot ids",
    )
    parser.add_argument("--linear-fitness-scale", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()] or None
    algorithms = [x.strip() for x in args.algorithms.split(",") if x.strip()] or None
    plot_ids = [x.strip() for x in args.plots.split(",") if x.strip()]

    cfg = PlotConfig(
        detailed_csv=Path(args.detailed_csv),
        summary_csv=Path(args.summary_csv),
        trajectory_csv=Path(args.trajectory_csv),
        output_dir=Path(args.output_dir),
        budgets=budgets,
        algorithms=algorithms,
        fig_format=args.fig_format,
        dpi=args.dpi,
        fitness_log_scale=not args.linear_fitness_scale,
        plot_ids=plot_ids,
    )
    paths = generate_plots(cfg)
    print(f"Generated {len(paths) - 1} plots in {cfg.output_dir}")


if __name__ == "__main__":
    main()
