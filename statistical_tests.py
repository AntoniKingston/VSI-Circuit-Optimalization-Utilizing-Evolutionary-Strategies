#!/usr/bin/env python3
"""Statistical hypothesis tests for VSI optimization experiment CSV outputs.

Designed for the analysis declared in the initial WdAE report (section 2):
  - Mann-Whitney U / Wilcoxon rank-sum for unpaired algorithm comparisons
  - Wilcoxon signed-rank for paired budget-scaling comparisons (same seed + config)
  - Friedman test for overall algorithm differences on matched seeds
  - Holm-Bonferroni correction for families of pairwise tests
  - Cliff's delta as a non-parametric effect size
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

ALGO_LABELS = {
    "es": "ES",
    "jde": "jDE",
    "cma": "CMA-ES",
    "pso": "PSO",
}


@dataclass
class StatTestConfig:
    detailed_csv: Path
    output_dir: Path
    budgets: List[int] = field(default_factory=lambda: [1000, 2000, 5000])
    algorithms: List[str] = field(default_factory=lambda: ["es", "jde", "cma", "pso"])
    tuning_budget: int = 1000
    alpha: float = 0.05
    # How to pick the representative hyperparameter set per algorithm.
    # "tuned_best": best median on tuning_budget, applied to all budgets
    # "per_budget_best": best median separately at each budget
    # "best_per_seed": min fitness across configs for each seed (oracle over grid)
    # "pooled": all successful runs for the algorithm at the budget
    config_mode: str = "tuned_best"
    test_ids: Optional[List[str]] = None
    include_runtime_tests: bool = True


@dataclass
class TestResult:
    test_id: str
    family: str
    hypothesis: str
    budget: Optional[int]
    algorithm_a: Optional[str]
    algorithm_b: Optional[str]
    config_mode: str
    n_a: int
    n_b: int
    median_a: float
    median_b: float
    statistic: float
    p_value: float
    p_holm: float
    significant_holm: bool
    effect_size: float
    effect_label: str
    notes: str = ""


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


def _median(values: Sequence[float]) -> float:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return math.nan
    return float(np.median(vals))


def cliffs_delta(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """Cliff's delta: P(X < Y) - P(X > Y) for X~A, Y~B. Negative => A tends to be lower (better)."""
    a = [v for v in sample_a if not math.isnan(v)]
    b = [v for v in sample_b if not math.isnan(v)]
    if not a or not b:
        return math.nan
    greater = sum(x > y for x in a for y in b)
    less = sum(x < y for x in a for y in b)
    n = len(a) * len(b)
    return float((less - greater) / n)


def _effect_label(delta: float) -> str:
    if math.isnan(delta):
        return "n/a"
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def holm_correction(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * m
    prev = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = min(1.0, (m - rank + 1) * p)
        adj = max(prev, adj)
        adjusted[idx] = adj
        prev = adj
    return adjusted


def _group_by_config(rows: Iterable[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["hyperparams_json"], []).append(row)
    return grouped


def _select_best_config(rows: Iterable[Dict[str, str]]) -> Optional[str]:
    by_cfg = _group_by_config(rows)
    best_key: Optional[str] = None
    best_median = math.inf
    for cfg, cfg_rows in by_cfg.items():
        vals = [_to_float(r["best_fitness"]) for r in cfg_rows if r["status"] == "ok"]
        if not vals:
            continue
        med = _median(vals)
        if med < best_median:
            best_median = med
            best_key = cfg
    return best_key


def _filter_rows(
    rows: List[Dict[str, str]],
    *,
    algorithm: str,
    budget: int,
    ok_only: bool = True,
) -> List[Dict[str, str]]:
    out = [
        r
        for r in rows
        if r["algorithm"] == algorithm and int(r["budget"]) == budget
    ]
    if ok_only:
        out = [r for r in out if r["status"] == "ok"]
    return out


def _resolve_config(
    rows: List[Dict[str, str]],
    algorithm: str,
    budget: int,
    cfg: StatTestConfig,
    tuned_configs: Dict[str, str],
) -> Optional[str]:
    if cfg.config_mode == "tuned_best":
        return tuned_configs.get(algorithm)
    if cfg.config_mode == "per_budget_best":
        return _select_best_config(_filter_rows(rows, algorithm=algorithm, budget=budget))
    return None


def _fitness_samples(
    rows: List[Dict[str, str]],
    algorithm: str,
    budget: int,
    cfg: StatTestConfig,
    tuned_configs: Dict[str, str],
) -> Tuple[List[float], str]:
    algo_rows = _filter_rows(rows, algorithm=algorithm, budget=budget)

    if cfg.config_mode in {"tuned_best", "per_budget_best"}:
        hp = _resolve_config(rows, algorithm, budget, cfg, tuned_configs)
        if hp is None:
            return [], ""
        vals = [
            _to_float(r["best_fitness"])
            for r in algo_rows
            if r["hyperparams_json"] == hp
        ]
        return vals, hp

    if cfg.config_mode == "best_per_seed":
        by_seed: Dict[str, float] = {}
        for r in algo_rows:
            seed = r["seed"]
            fit = _to_float(r["best_fitness"])
            by_seed[seed] = min(by_seed.get(seed, math.inf), fit)
        vals = [v for v in by_seed.values() if not math.isnan(v)]
        return vals, "best_over_grid_per_seed"

    if cfg.config_mode == "pooled":
        vals = [_to_float(r["best_fitness"]) for r in algo_rows]
        return vals, "all_configs_pooled"

    raise ValueError(f"Unknown config_mode: {cfg.config_mode}")


def _paired_budget_samples(
    rows: List[Dict[str, str]],
    algorithm: str,
    budget_low: int,
    budget_high: int,
    hyperparams_json: str,
) -> Tuple[List[float], List[float], int]:
    low_map = {
        r["seed"]: _to_float(r["best_fitness"])
        for r in _filter_rows(rows, algorithm=algorithm, budget=budget_low)
        if r["hyperparams_json"] == hyperparams_json
    }
    high_map = {
        r["seed"]: _to_float(r["best_fitness"])
        for r in _filter_rows(rows, algorithm=algorithm, budget=budget_high)
        if r["hyperparams_json"] == hyperparams_json
    }
    common = sorted(set(low_map) & set(high_map))
    low_vals = [low_map[s] for s in common]
    high_vals = [high_map[s] for s in common]
    return low_vals, high_vals, len(common)


def _compute_tuned_configs(
    rows: List[Dict[str, str]],
    cfg: StatTestConfig,
) -> Dict[str, str]:
    tuned: Dict[str, str] = {}
    for algo in cfg.algorithms:
        tuning_rows = _filter_rows(rows, algorithm=algo, budget=cfg.tuning_budget)
        best = _select_best_config(tuning_rows)
        if best is not None:
            tuned[algo] = best
    return tuned


def _mann_whitney_result(
    *,
    test_id: str,
    family: str,
    hypothesis: str,
    budget: int,
    algo_a: str,
    algo_b: str,
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    cfg: StatTestConfig,
    notes: str = "",
) -> TestResult:
    med_a = _median(sample_a)
    med_b = _median(sample_b)
    if len(sample_a) < 1 or len(sample_b) < 1:
        return TestResult(
            test_id=test_id,
            family=family,
            hypothesis=hypothesis,
            budget=budget,
            algorithm_a=algo_a,
            algorithm_b=algo_b,
            config_mode=cfg.config_mode,
            n_a=len(sample_a),
            n_b=len(sample_b),
            median_a=med_a,
            median_b=med_b,
            statistic=math.nan,
            p_value=math.nan,
            p_holm=math.nan,
            significant_holm=False,
            effect_size=math.nan,
            effect_label="n/a",
            notes=notes or "insufficient samples",
        )

    # One-sided: H1 states algorithm_a achieves lower (better) fitness than algorithm_b.
    stat, p_two = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")
    _, p_less = stats.mannwhitneyu(sample_a, sample_b, alternative="less")
    delta = cliffs_delta(sample_a, sample_b)
    return TestResult(
        test_id=test_id,
        family=family,
        hypothesis=hypothesis,
        budget=budget,
        algorithm_a=algo_a,
        algorithm_b=algo_b,
        config_mode=cfg.config_mode,
        n_a=len(sample_a),
        n_b=len(sample_b),
        median_a=med_a,
        median_b=med_b,
        statistic=float(stat),
        p_value=float(p_two),
        p_holm=math.nan,
        significant_holm=False,
        effect_size=delta,
        effect_label=_effect_label(delta),
        notes=notes + f"; p_one_sided_a_better={p_less:.6g}",
    )


def test_algorithm_pairwise(cfg: StatTestConfig, rows: List[Dict[str, str]]) -> List[TestResult]:
    """H1: At fixed FES budget, final fitness distributions differ between algorithms."""
    tuned = _compute_tuned_configs(rows, cfg)
    results: List[TestResult] = []

    for budget in cfg.budgets:
        raw: List[TestResult] = []
        for algo_a, algo_b in combinations(cfg.algorithms, 2):
            sample_a, cfg_a = _fitness_samples(rows, algo_a, budget, cfg, tuned)
            sample_b, cfg_b = _fitness_samples(rows, algo_b, budget, cfg, tuned)
            notes = f"config_a={cfg_a}; config_b={cfg_b}"
            raw.append(
                _mann_whitney_result(
                    test_id="algorithm_pairwise_mw",
                    family=f"algorithm_pairwise_budget_{budget}",
                    hypothesis=(
                        f"H0: median fitness({ALGO_LABELS.get(algo_a, algo_a)}) = "
                        f"median fitness({ALGO_LABELS.get(algo_b, algo_b)}) at FES={budget}"
                    ),
                    budget=budget,
                    algo_a=algo_a,
                    algo_b=algo_b,
                    sample_a=sample_a,
                    sample_b=sample_b,
                    cfg=cfg,
                    notes=notes,
                )
            )
        holm = holm_correction([r.p_value if not math.isnan(r.p_value) else 1.0 for r in raw])
        for r, p_adj in zip(raw, holm):
            r.p_holm = p_adj
            r.significant_holm = p_adj < cfg.alpha
        results.extend(raw)
    return results


def test_algorithm_vs_pso(cfg: StatTestConfig, rows: List[Dict[str, str]]) -> List[TestResult]:
    """H2: Each evolutionary method outperforms the PSO baseline at the same FES budget."""
    if "pso" not in cfg.algorithms:
        return []
    tuned = _compute_tuned_configs(rows, cfg)
    baselines = [a for a in cfg.algorithms if a != "pso"]
    results: List[TestResult] = []

    for budget in cfg.budgets:
        raw: List[TestResult] = []
        for algo in baselines:
            sample_a, cfg_a = _fitness_samples(rows, algo, budget, cfg, tuned)
            sample_b, cfg_b = _fitness_samples(rows, "pso", budget, cfg, tuned)
            raw.append(
                _mann_whitney_result(
                    test_id="algorithm_vs_pso_mw",
                    family=f"algorithm_vs_pso_budget_{budget}",
                    hypothesis=(
                        f"H0: {ALGO_LABELS.get(algo, algo)} is not better than PSO at FES={budget}"
                    ),
                    budget=budget,
                    algo_a=algo,
                    algo_b="pso",
                    sample_a=sample_a,
                    sample_b=sample_b,
                    cfg=cfg,
                    notes=f"config_a={cfg_a}; config_pso={cfg_b}; one-sided: A better than PSO",
                )
            )
        holm = holm_correction([r.p_value if not math.isnan(r.p_value) else 1.0 for r in raw])
        for r, p_adj in zip(raw, holm):
            r.p_holm = p_adj
            r.significant_holm = p_adj < cfg.alpha
        results.extend(raw)
    return results


def test_budget_scaling(cfg: StatTestConfig, rows: List[Dict[str, str]]) -> List[TestResult]:
    """H3: Increasing FES budget improves final fitness for the tuned best configuration."""
    tuned = _compute_tuned_configs(rows, cfg)
    results: List[TestResult] = []
    budget_pairs = list(zip(cfg.budgets[:-1], cfg.budgets[1:]))

    for algo in cfg.algorithms:
        hp = tuned.get(algo)
        if hp is None:
            continue
        for budget_low, budget_high in budget_pairs:
            low_vals, high_vals, n_pairs = _paired_budget_samples(
                rows, algo, budget_low, budget_high, hp
            )
            if n_pairs < 2:
                results.append(
                    TestResult(
                        test_id="budget_scaling_wilcoxon",
                        family=f"budget_scaling_{algo}",
                        hypothesis=(
                            f"H0: FES={budget_low} and FES={budget_high} yield equal fitness "
                            f"for {ALGO_LABELS.get(algo, algo)} (paired Wilcoxon)"
                        ),
                        budget=budget_high,
                        algorithm_a=algo,
                        algorithm_b=None,
                        config_mode=cfg.config_mode,
                        n_a=n_pairs,
                        n_b=n_pairs,
                        median_a=_median(low_vals),
                        median_b=_median(high_vals),
                        statistic=math.nan,
                        p_value=math.nan,
                        p_holm=math.nan,
                        significant_holm=False,
                        effect_size=math.nan,
                        effect_label="n/a",
                        notes=f"config={hp}; insufficient paired samples (n={n_pairs})",
                    )
                )
                continue

            # One-sided: H1 higher budget yields lower (better) fitness.
            stat, p_two = stats.wilcoxon(low_vals, high_vals, alternative="two-sided")
            _, p_less = stats.wilcoxon(low_vals, high_vals, alternative="less")
            diff = np.array(high_vals) - np.array(low_vals)
            results.append(
                TestResult(
                    test_id="budget_scaling_wilcoxon",
                    family=f"budget_scaling_{algo}",
                    hypothesis=(
                        f"H0: FES={budget_low} and FES={budget_high} yield equal fitness "
                        f"for {ALGO_LABELS.get(algo, algo)} (paired Wilcoxon)"
                    ),
                    budget=budget_high,
                    algorithm_a=algo,
                    algorithm_b=None,
                    config_mode=cfg.config_mode,
                    n_a=n_pairs,
                    n_b=n_pairs,
                    median_a=_median(low_vals),
                    median_b=_median(high_vals),
                    statistic=float(stat),
                    p_value=float(p_two),
                    p_holm=math.nan,
                    significant_holm=bool(p_two < cfg.alpha),
                    effect_size=float(np.median(diff)),
                    effect_label="median_diff_high_minus_low",
                    notes=(
                        f"config={hp}; paired_seeds={n_pairs}; "
                        f"p_one_sided_high_budget_better={p_less:.6g}"
                    ),
                )
            )
    return results


def test_friedman_over_algorithms(cfg: StatTestConfig, rows: List[Dict[str, str]]) -> List[TestResult]:
    """H4: At fixed FES budget, all algorithms perform equally (Friedman on matched seeds)."""
    tuned = _compute_tuned_configs(rows, cfg)
    results: List[TestResult] = []

    for budget in cfg.budgets:
        per_algo_seed: Dict[str, Dict[str, float]] = {}
        for algo in cfg.algorithms:
            hp = tuned.get(algo)
            if hp is None:
                continue
            per_algo_seed[algo] = {
                r["seed"]: _to_float(r["best_fitness"])
                for r in _filter_rows(rows, algorithm=algo, budget=budget)
                if r["hyperparams_json"] == hp
            }

        if len(per_algo_seed) < 3:
            continue

        common_seeds = set.intersection(*(set(m) for m in per_algo_seed.values()))
        if len(common_seeds) < 2:
            results.append(
                TestResult(
                    test_id="algorithm_friedman",
                    family=f"algorithm_friedman_budget_{budget}",
                    hypothesis=(
                        f"H0: all algorithms have equal fitness at FES={budget} "
                        "(Friedman test on matched seeds)"
                    ),
                    budget=budget,
                    algorithm_a=None,
                    algorithm_b=None,
                    config_mode=cfg.config_mode,
                    n_a=len(common_seeds),
                    n_b=len(cfg.algorithms),
                    median_a=math.nan,
                    median_b=math.nan,
                    statistic=math.nan,
                    p_value=math.nan,
                    p_holm=math.nan,
                    significant_holm=False,
                    effect_size=math.nan,
                    effect_label="n/a",
                    notes=f"insufficient matched seeds (n={len(common_seeds)})",
                )
            )
            continue

        ordered_seeds = sorted(common_seeds)
        samples = [
            [per_algo_seed[algo][seed] for seed in ordered_seeds]
            for algo in cfg.algorithms
            if algo in per_algo_seed
        ]
        stat, p_value = stats.friedmanchisquare(*samples)
        results.append(
            TestResult(
                test_id="algorithm_friedman",
                family=f"algorithm_friedman_budget_{budget}",
                hypothesis=(
                    f"H0: all algorithms have equal fitness at FES={budget} "
                    "(Friedman test on matched seeds)"
                ),
                budget=budget,
                algorithm_a=None,
                algorithm_b=None,
                config_mode=cfg.config_mode,
                n_a=len(ordered_seeds),
                n_b=len(samples),
                median_a=math.nan,
                median_b=math.nan,
                statistic=float(stat),
                p_value=float(p_value),
                p_holm=math.nan,
                significant_holm=bool(p_value < cfg.alpha),
                effect_size=math.nan,
                effect_label="n/a",
                notes=f"matched_seeds={len(ordered_seeds)}; tuned_configs={json.dumps(tuned)}",
            )
        )
    return results


def test_runtime_pairwise(cfg: StatTestConfig, rows: List[Dict[str, str]]) -> List[TestResult]:
    """H5: Wall-clock runtime distributions differ between algorithms (secondary metric)."""
    tuned = _compute_tuned_configs(rows, cfg)
    results: List[TestResult] = []

    for budget in cfg.budgets:
        raw: List[TestResult] = []
        for algo_a, algo_b in combinations(cfg.algorithms, 2):
            rows_a = _filter_rows(rows, algorithm=algo_a, budget=budget)
            rows_b = _filter_rows(rows, algorithm=algo_b, budget=budget)
            hp_a = _resolve_config(rows, algo_a, budget, cfg, tuned)
            hp_b = _resolve_config(rows, algo_b, budget, cfg, tuned)
            if hp_a:
                rows_a = [r for r in rows_a if r["hyperparams_json"] == hp_a]
            if hp_b:
                rows_b = [r for r in rows_b if r["hyperparams_json"] == hp_b]
            sample_a = [_to_float(r["runtime_s"]) for r in rows_a]
            sample_b = [_to_float(r["runtime_s"]) for r in rows_b]
            raw.append(
                _mann_whitney_result(
                    test_id="runtime_pairwise_mw",
                    family=f"runtime_pairwise_budget_{budget}",
                    hypothesis=(
                        f"H0: runtime({ALGO_LABELS.get(algo_a, algo_a)}) = "
                        f"runtime({ALGO_LABELS.get(algo_b, algo_b)}) at FES={budget}"
                    ),
                    budget=budget,
                    algo_a=algo_a,
                    algo_b=algo_b,
                    sample_a=sample_a,
                    sample_b=sample_b,
                    cfg=cfg,
                    notes="metric=runtime_s; two-sided (no direction assumed)",
                )
            )
        holm = holm_correction([r.p_value if not math.isnan(r.p_value) else 1.0 for r in raw])
        for r, p_adj in zip(raw, holm):
            r.p_holm = p_adj
            r.significant_holm = p_adj < cfg.alpha
        results.extend(raw)
    return results


TEST_REGISTRY = {
    "algorithm_pairwise_mw": test_algorithm_pairwise,
    "algorithm_vs_pso_mw": test_algorithm_vs_pso,
    "budget_scaling_wilcoxon": test_budget_scaling,
    "algorithm_friedman": test_friedman_over_algorithms,
    "runtime_pairwise_mw": test_runtime_pairwise,
}


def _result_to_row(result: TestResult) -> Dict[str, Any]:
    return {
        "test_id": result.test_id,
        "family": result.family,
        "hypothesis": result.hypothesis,
        "budget": result.budget if result.budget is not None else "",
        "algorithm_a": result.algorithm_a or "",
        "algorithm_b": result.algorithm_b or "",
        "config_mode": result.config_mode,
        "n_a": result.n_a,
        "n_b": result.n_b,
        "median_a": result.median_a,
        "median_b": result.median_b,
        "statistic": result.statistic,
        "p_value": result.p_value,
        "p_holm": result.p_holm,
        "significant_holm": int(result.significant_holm),
        "effect_size": result.effect_size,
        "effect_label": result.effect_label,
        "notes": result.notes,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, cfg: StatTestConfig, results: List[TestResult], tuned: Dict[str, str]) -> None:
    lines = [
        "# Statistical test report",
        "",
        f"Input CSV: `{cfg.detailed_csv}`",
        f"Config mode: `{cfg.config_mode}` (tuning budget: {cfg.tuning_budget})",
        f"Significance level: alpha = {cfg.alpha}",
        "",
        "## Hypotheses tested",
        "",
        "1. **Algorithm comparison (Mann-Whitney U, Holm-corrected)**",
        "   H0: At a fixed FES budget, two algorithms reach the same final fitness distribution.",
        "",
        "2. **Evolutionary methods vs PSO baseline (Mann-Whitney U, Holm-corrected)**",
        "   H0: ES/jDE/CMA-ES are not better than PSO at the same FES budget.",
        "",
        "3. **Budget scaling (paired Wilcoxon signed-rank)**",
        "   H0: For the tuned best configuration, increasing FES does not improve fitness.",
        "",
        "4. **Global algorithm comparison (Friedman test on matched seeds)**",
        "   H0: All algorithms perform equally at the same FES budget.",
        "",
        "5. **Runtime comparison (Mann-Whitney U, Holm-corrected)**",
        "   H0: Wall-clock runtimes are equal between two algorithms.",
        "",
        "## Tuned best configurations (selected on FES={})".format(cfg.tuning_budget),
        "",
    ]
    for algo, hp in tuned.items():
        label = ALGO_LABELS.get(algo, algo)
        lines.append(f"- **{label}**: `{hp}`")
    lines.extend(["", "## Significant results (Holm-adjusted where applicable)", ""])

    sig = [r for r in results if r.significant_holm or (math.isnan(r.p_holm) and r.p_value < cfg.alpha)]
    if not sig:
        lines.append("_No significant results at alpha={}._".format(cfg.alpha))
    else:
        for r in sig:
            a = ALGO_LABELS.get(r.algorithm_a or "", r.algorithm_a or "-")
            b = ALGO_LABELS.get(r.algorithm_b or "", r.algorithm_b or "-")
            p_disp = r.p_holm if not math.isnan(r.p_holm) else r.p_value
            lines.append(
                f"- `{r.test_id}` budget={r.budget} {a} vs {b}: "
                f"p={r.p_value:.4g}, p_Holm={p_disp:.4g}, "
                f"median_a={r.median_a:.6g}, median_b={r.median_b:.6g}, "
                f"Cliff's d={r.effect_size:.3f} ({r.effect_label})"
            )

    lines.extend(["", "## All test rows", "", "See `statistical_tests_all.csv` for the full table.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_statistical_tests(cfg: StatTestConfig) -> List[TestResult]:
    rows = _read_csv(cfg.detailed_csv)
    tuned = _compute_tuned_configs(rows, cfg)

    test_ids = cfg.test_ids or list(TEST_REGISTRY.keys())
    if not cfg.include_runtime_tests and "runtime_pairwise_mw" in test_ids:
        test_ids = [t for t in test_ids if t != "runtime_pairwise_mw"]

    all_results: List[TestResult] = []
    for test_id in test_ids:
        if test_id not in TEST_REGISTRY:
            raise ValueError(f"Unknown test_id: {test_id}")
        all_results.extend(TEST_REGISTRY[test_id](cfg, rows))

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = [_result_to_row(r) for r in all_results]
    _write_csv(out_dir / "statistical_tests_all.csv", csv_rows)

    by_test: Dict[str, List[Dict[str, Any]]] = {}
    for row in csv_rows:
        by_test.setdefault(row["test_id"], []).append(row)
    for test_id, test_rows in by_test.items():
        _write_csv(out_dir / f"{test_id}.csv", test_rows)

    _write_report(out_dir / "statistical_report.md", cfg, all_results, tuned)
    return all_results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run statistical tests on experiment CSV results.")
    parser.add_argument(
        "--detailed-csv",
        type=Path,
        default=Path("results/detailed_results_merged.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/statistics"))
    parser.add_argument("--budgets", type=str, default="1000,2000,5000")
    parser.add_argument("--algorithms", type=str, default="es,jde,cma,pso")
    parser.add_argument("--tuning-budget", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--config-mode",
        choices=["tuned_best", "per_budget_best", "best_per_seed", "pooled"],
        default="tuned_best",
    )
    parser.add_argument("--tests", type=str, default="")
    parser.add_argument("--skip-runtime", action="store_true")
    args = parser.parse_args()

    test_ids = [t.strip() for t in args.tests.split(",") if t.strip()] or None
    cfg = StatTestConfig(
        detailed_csv=args.detailed_csv,
        output_dir=args.output_dir,
        budgets=[int(x) for x in args.budgets.split(",") if x.strip()],
        algorithms=[x.strip() for x in args.algorithms.split(",") if x.strip()],
        tuning_budget=args.tuning_budget,
        alpha=args.alpha,
        config_mode=args.config_mode,
        test_ids=test_ids,
        include_runtime_tests=not args.skip_runtime,
    )
    results = run_statistical_tests(cfg)
    print(f"Statistical tests finished. {len(results)} rows written to {cfg.output_dir}")


if __name__ == "__main__":
    main()
