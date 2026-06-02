#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_csv_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_float_safe(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def _is_ok_with_fitness(row: Dict[str, str]) -> bool:
    if row.get("status", "") != "ok":
        return False
    val = _to_float_safe(row.get("best_fitness", "nan"))
    return not math.isnan(val)


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] + (values[hi] - values[lo]) * (idx - lo))


def _build_summary(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("algorithm", ""),
            row.get("budget", ""),
            row.get("hyperparams_json", ""),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, str]] = []
    for (algorithm, budget, hyperparams_json), group_rows in grouped.items():
        vals = [_to_float_safe(r["best_fitness"]) for r in group_rows if _is_ok_with_fitness(r)]
        vals = [v for v in vals if not math.isnan(v)]
        summary_rows.append(
            {
                "algorithm": algorithm,
                "budget": budget,
                "hyperparams_json": hyperparams_json,
                "n_runs": str(len(group_rows)),
                "n_ok": str(len(vals)),
                "mean": str(float(statistics.mean(vals)) if vals else math.nan),
                "median": str(float(statistics.median(vals)) if vals else math.nan),
                "std": str(float(statistics.stdev(vals)) if len(vals) > 1 else 0.0),
                "min": str(float(min(vals)) if vals else math.nan),
                "max": str(float(max(vals)) if vals else math.nan),
                "q1": str(_quantile(vals, 0.25)),
                "q3": str(_quantile(vals, 0.75)),
            }
        )
    return summary_rows


def _merge_detailed(base_rows: List[Dict[str, str]], pso_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # Remove old PSO rows from base, then append fresh PSO rows.
    merged = [r for r in base_rows if r.get("algorithm") != "pso"]
    merged.extend(pso_rows)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge PSO rerun results into main experiment CSVs.")
    parser.add_argument("--results-dir", default="results", help="Directory with result CSV files.")
    parser.add_argument("--base-detailed", default="detailed_results.csv", help="Main detailed CSV filename.")
    parser.add_argument("--pso-detailed", default="pso_detailed_results.csv", help="PSO-only detailed CSV filename.")
    parser.add_argument("--out-detailed", default="detailed_results_merged.csv", help="Merged detailed CSV filename.")
    parser.add_argument("--out-summary", default="summary_results_merged.csv", help="Merged summary CSV filename.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    base_detailed_path = results_dir / args.base_detailed
    pso_detailed_path = results_dir / args.pso_detailed
    out_detailed_path = results_dir / args.out_detailed
    out_summary_path = results_dir / args.out_summary

    base_rows = _read_csv_rows(base_detailed_path)
    pso_rows = _read_csv_rows(pso_detailed_path)

    if not base_rows and not pso_rows:
        raise RuntimeError("Both input detailed CSVs are empty.")

    # Normalize columns across rows (union of fieldnames).
    fieldnames = []
    for rows in (base_rows, pso_rows):
        if rows:
            for key in rows[0].keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    for key in ["algorithm", "budget", "seed", "hyperparams_json", "status", "best_fitness", "best_solution_json", "fevals", "runtime_s", "error"]:
        if key not in fieldnames:
            fieldnames.append(key)

    merged_rows = _merge_detailed(base_rows, pso_rows)
    # Ensure all rows have all columns.
    normalized_rows: List[Dict[str, str]] = []
    for row in merged_rows:
        normalized_rows.append({k: row.get(k, "") for k in fieldnames})

    _write_csv_rows(out_detailed_path, normalized_rows, fieldnames)

    summary_rows = _build_summary(normalized_rows)
    summary_fieldnames = [
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
    ]
    _write_csv_rows(out_summary_path, summary_rows, summary_fieldnames)

    print(f"Merged detailed results: {out_detailed_path}")
    print(f"Merged summary results: {out_summary_path}")
    print(f"Rows: base={len(base_rows)} | pso={len(pso_rows)} | merged={len(normalized_rows)}")


if __name__ == "__main__":
    main()
