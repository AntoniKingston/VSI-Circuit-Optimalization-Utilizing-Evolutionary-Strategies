#!/usr/bin/env python3
"""Example runner: statistical tests for the VSI optimization report."""

import math
from pathlib import Path

from statistical_tests import StatTestConfig, run_statistical_tests


def main() -> None:
    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results"
    stats_dir = results_dir / "statistics"

    primary_cfg = StatTestConfig(
        detailed_csv=results_dir / "detailed_results.csv",
        output_dir=stats_dir / "tuned_best",
        budgets=[50, 100, 200],
        algorithms=["es", "jde", "cma", "pso"],
        tuning_budget=50,
        alpha=0.05,
        config_mode="tuned_best",
        test_ids=[
            "algorithm_pairwise_mw",
            "algorithm_vs_pso_mw",
            "budget_scaling_wilcoxon",
            "algorithm_friedman",
            "runtime_pairwise_mw",
        ],
    )
    primary_results = run_statistical_tests(primary_cfg)

    oracle_cfg = StatTestConfig(
        detailed_csv=results_dir / "detailed_results.csv",
        output_dir=stats_dir / "best_per_seed",
        budgets=[50, 100, 200],
        algorithms=["es", "jde", "cma", "pso"],
        tuning_budget=50,
        alpha=0.05,
        config_mode="best_per_seed",
        test_ids=["algorithm_pairwise_mw", "algorithm_vs_pso_mw", "algorithm_friedman"],
        include_runtime_tests=False,
    )
    run_statistical_tests(oracle_cfg)

    sig = [
        r
        for r in primary_results
        if r.significant_holm
        or (math.isnan(r.p_holm) and r.p_value < primary_cfg.alpha)
    ]
    print("Report statistics finished.")
    print(f"Primary output: {stats_dir / 'tuned_best'}")
    print(f"Supplementary output: {stats_dir / 'best_per_seed'}")
    print(f"Significant primary tests (alpha={primary_cfg.alpha}): {len(sig)}")


if __name__ == "__main__":
    main()
