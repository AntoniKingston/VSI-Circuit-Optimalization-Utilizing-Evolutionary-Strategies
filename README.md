# VSI Circuit Optimization with Evolutionary Strategies

**Repository:** [github.com/AntoniKingston/VSI-Circuit-Optimalization-Utilizing-Evolutionary-Strategies](https://github.com/AntoniKingston/VSI-Circuit-Optimalization-Utilizing-Evolutionary-Strategies)

WdAE project (task 24): tune DLQR weighting parameters of a voltage-source inverter (VSI) simulator using derivative-free optimizers. The fitness function is a black-box MATLAB evaluation exposed to Python via XML-RPC (ES, jDE, CMA-ES) or called natively (PSO).

**Authors:** Antoni Kingston, Marcel Wronkowski  
**Report:** `report/report.tex` (compile with `pdflatex` in `report/`)

---

## Repository layout

| Path | Role |
|------|------|
| `algorithms.py` | ES, jDE, CMA-ES, PSO wrapper, `compute_fitness` |
| `client.py` | XML-RPC client used by Python optimizers |
| `server.py` | XML-RPC server wrapping the MATLAB simulator |
| `run_experiments.py` | Main batch experiment runner |
| `experiments_config.txt` | Grids, budgets, seeds, output paths |
| `plot_results.py` / `generate_report_plots.py` | Figures for the report |
| `statistical_tests.py` / `run_report_statistics.py` | Mann–Whitney, Wilcoxon, Friedman + Holm |
| `src/matlab/` | VSI plant, DLQR evaluator, PSO optimizer |
| `results/` | CSV outputs, figures, statistics (generated) |
| `report/` | LaTeX final report |

Legacy helpers `run_pso_only.py` and `merge_results.py` exist for old runs where PSO failed; **they are not needed** if you use the current PSO integration (see below).

---

## Requirements

- **Linux** (tested on Arch/CachyOS)
- **Python 3.12+**
- **MATLAB R2025a** with **MATLAB Engine API for Python** (`matlabengine`)
- Python packages: see `requirements.txt` (`numpy`, `cma`, `scipy`, `matplotlib`, …)

### Python environment

```bash
git clone https://github.com/AntoniKingston/VSI-Circuit-Optimalization-Utilizing-Evolutionary-Strategies.git
cd VSI-Circuit-Optimalization-Utilizing-Evolutionary-Strategies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## MATLAB setup

PSO connects to a **shared MATLAB engine** named `ubuntu_matlab` (configurable in `PSOMatlabConfig`).

1. Start MATLAB (optionally inside distrobox — see `instrukcje.txt`):

   ```bash
   # example from project notes
   distrobox enter matlab-ubuntu -- bash --noprofile --norc
   ~/matlab/bin/matlab -nodesktop -nosplash
   ```

2. In MATLAB, share the engine:

   ```matlab
   matlab.engine.shareEngine('ubuntu_matlab')
   ```

Keep this session running while experiments include PSO.

---

## How to reproduce the reported results

The pipeline has four stages: **simulator server → experiments → analysis → report PDF**.

### 1. Start the XML-RPC simulator (ES / jDE / CMA-ES)

In a dedicated terminal (with venv activated):

```bash
source .venv/bin/activate
python server.py
# optional: python server.py -v
```

Default listen address: `127.0.0.1:8484`.  
ES, jDE, and CMA-ES call `client.py`, which forwards candidate vectors to this server.

### 2. Run the experiment grid

Ensure the shared MATLAB engine is up if `algorithms` includes `pso` (see `experiments_config.txt`).

In another terminal:

```bash
source .venv/bin/activate
python run_experiments.py experiments_config.txt
```

This writes under `results/`:

- `detailed_results.csv` — one row per run (algorithm, budget, seed, hyperparameters, fitness, runtime, …)
- `summary_results.csv` — aggregated stats per configuration
- `trajectories.csv` — best-so-far curves (if `save_trajectories = 1`)
- `progress.json` — live progress
- `run_debug.txt` — log

Current default config (`experiments_config.txt`):

- Algorithms: `es,jde,cma,pso`
- FES budgets: `50,100,200`
- Seeds: `1,2,3,4,5`
- PSO runs **serially** (`parallel_pso = 0`) to avoid contention on the shared engine

**Note:** A full grid takes substantial wall-clock time (thousands of simulator calls). For a smoke test, temporarily reduce grids/seeds/budgets in `experiments_config.txt`.

**PSO:** No merge step is required. `run_experiments.py` passes `c1`, `c2`, `rang_coef`, and `random_seed` through `PSOMatlabConfig` into `src/matlab/main.m`. You do **not** need `run_pso_only.py` or `merge_results.py` unless you are repairing old CSVs from a broken PSO run.

### 3. Generate figures and statistics

```bash
source .venv/bin/activate
python generate_report_plots.py
python run_report_statistics.py
```

Outputs:

- `results/figures/*.pdf` — boxplots, ECDF, budget scaling, runtime, sensitivity plots
- `results/statistics/tuned_best/` — Holm-corrected tests (`statistical_report.md`, CSV tables)

Statistics use **tuned-best** configurations selected on budget `B=50` (see `statistical_tests.py` / report Section 5).

Figures are included from `../results/figures/`.

---

## Quick sanity checks

**XML-RPC + one fitness evaluation:**

```bash
python server.py   # terminal 1
python client.py -d -2.5 -1.0 0.5 1.2   # terminal 2
```

**PSO only (requires shared engine):**

```bash
python -c "
from algorithms import PSOMatlabConfig, optimize_pso_matlab
cfg = PSOMatlabConfig(max_iter=5, particle_number=20, random_seed=1, echo_logs=True)
print(optimize_pso_matlab(cfg)[0])
"
```

---

## Configuration reference

Edit `experiments_config.txt` to change:

- `[general]`: `algorithms`, `budgets`, `seeds`, `max_workers`, `parallel_pso`, output file names
- `[grid.es]`, `[grid.jde]`, `[grid.cma]`, `[grid.pso]`: hyperparameter grids

FES mapping per algorithm is implemented in `run_experiments.py` (Section 3 of the report): e.g. ES generations \(G=\lfloor B/\lambda\rfloor\), PSO \(G=\lfloor B/N_{\mathrm{part}}\rfloor\).

---

## Course deliverables checklist

- [ ] GitHub repository (or `git bundle` with full history for course submission)
- [ ] `results/detailed_results.csv` + seeds listed in config
- [ ] `report/report.pdf`
- [ ] This `README.md` for reproduction
- [ ] Optional: archive `results/` if larger than 5 MB (per WdAE instructions)

---

## References

- Initial task specification: `WdAE.pdf` (WdAE course, task 24)
- jDE: Brest et al., IEEE TEVC 2006
- CMA-ES: Hansen & Ostermeier, Evolutionary Computation 2001
