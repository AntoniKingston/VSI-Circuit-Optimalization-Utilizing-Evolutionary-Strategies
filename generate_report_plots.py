#!/usr/bin/env python3
"""Example runner: generate report-ready plots for the VSI optimization study."""

from pathlib import Path

from plot_results import PlotConfig, generate_plots


def main() -> None:
    project_root = Path(__file__).resolve().parent
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"

    # Core report figures for Springer Nature Results section.
    core_plot_ids = [
        "algorithm_fitness_boxplot",
        "budget_scaling",
        "success_rate",
        "runtime_boxplot",
        "ecdf_fitness",
        "best_config_ranking",
        "hyperparam_sensitivity",
    ]

    core_cfg = PlotConfig(
        detailed_csv=results_dir / "detailed_results.csv",
        summary_csv=results_dir / "summary_results.csv",
        trajectory_csv=results_dir / "trajectories.csv",
        output_dir=figures_dir,
        budgets=[50, 100, 200],
        algorithms=["es", "jde", "cma", "pso"],
        fig_format="pdf",
        dpi=300,
        fitness_log_scale=True,
        plot_ids=core_plot_ids,
    )
    core_paths = generate_plots(core_cfg)

    # Optional convergence plots if trajectory CSV exists.
    if (results_dir / "trajectories.csv").exists():
        conv_cfg = PlotConfig(
            detailed_csv=results_dir / "detailed_results.csv",
            summary_csv=results_dir / "summary_results.csv",
            trajectory_csv=results_dir / "trajectories.csv",
            output_dir=figures_dir / "convergence",
            budgets=[50, 100, 200],
            algorithms=["es", "jde", "cma", "pso"],
            fig_format="pdf",
            dpi=300,
            plot_ids=["convergence_curves"],
        )
        generate_plots(conv_cfg)

    print("Report plot generation finished.")
    print(f"Output directory: {figures_dir}")
    print(f"Generated files: {len(core_paths) - 1}")


if __name__ == "__main__":
    main()
