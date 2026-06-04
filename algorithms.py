from typing import List, Callable, Tuple, Optional, Dict, Any
import subprocess
from dataclasses import dataclass, field
import numpy as np
from enum import Enum
import re
from pathlib import Path
import io
import sys
try:
    import cma
except ImportError:
    cma = None
try:
    import matlab.engine
except ImportError:
    matlab = None

# Function works only if server is already running
def compute_fitness(individual: List[float], opt_domain_pos_len: int = 4, client_script_name: str = "client.py") -> float:
    individual = individual[:opt_domain_pos_len]
    cmd = ["python", client_script_name, "-d", *map(str, individual)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    out_str = result.stdout.strip()
    err_str = result.stderr.strip()
    match = re.search(r"Objective function value = ([^\s]+)", out_str)
    if result.returncode != 0 or match is None:
        raise RuntimeError(
            "Failed to evaluate fitness via XML-RPC client. "
            f"cmd={' '.join(cmd)} | returncode={result.returncode} | "
            f"stdout={out_str!r} | stderr={err_str!r}"
        )
    return float(match.group(1))

def initiate_population(n_individuals: int, initial_dist: Callable[[], List[float]], initial_hyperparam_values: List[float]) -> List[List[float]]:
    population = []
    for _ in range(n_individuals):
        individual = initial_dist()
        individual += initial_hyperparam_values
        population.append(individual)
    return population

def crossover(parent_1: List[float], parent_2: List[float]) -> List[float]:
    assert len(parent_1) == len(parent_2)
    crossover_point = np.random.randint(1, len(parent_1) - 1)
    child = parent_1[:crossover_point] + parent_2[crossover_point:]
    return child

def mutate(
    individual: List[float],
    mutation_dist: Callable[[List, Dict[str, int]], Callable[[], List[float]]],
    hyperparam_mutation_dists: List[Callable[[], float]],
    hyperparam_name_to_id_dict: Dict[str, int],
    search_space_limits: Optional[List[Tuple[float, float]]] = None,
) -> List[float]:
    opt_domain_pos_len = len(individual) - len(hyperparam_mutation_dists)
    opt_domain = individual[:opt_domain_pos_len]
    hyperparam_domain = individual[opt_domain_pos_len:]
    mutation_dist_f = mutation_dist(hyperparam_domain, hyperparam_name_to_id_dict)
    opt_domain = list(np.asarray(opt_domain) + np.asarray(mutation_dist_f()))
    if search_space_limits:
        opt_domain = _clip_to_bounds(opt_domain, search_space_limits)
    for i, hyperparam_mutation_dist in enumerate(hyperparam_mutation_dists):
        if hyperparam_name_to_id_dict.get("sigma") == i:
            hyperparam_domain[i] *= float(hyperparam_mutation_dist())
            hyperparam_domain[i] = float(np.clip(hyperparam_domain[i], 1e-8, 10.0))
        else:
            hyperparam_domain[i] += float(hyperparam_mutation_dist())
    return list(opt_domain) + list(hyperparam_domain)

# We allow for no crossover (crossover between identical parents)
def create_children(
    parents: List[List[float]],
    n_children: int,
    mutation_dist: Callable[[List, Dict[str, int]], Callable[[], List[float]]],
    hyperparam_mutation_dists: List[Callable[[], float]],
    hyperparam_name_to_id_dict: Dict[str, int],
    search_space_limits: Optional[List[Tuple[float, float]]] = None,
) -> List[List[float]]:
    children = []
    for _ in range(n_children):
        id_1, id_2 = np.random.choice(len(parents), 2, replace=False)
        parent_1 = parents[id_1]
        parent_2 = parents[id_2]
        child = crossover(parent_1, parent_2) if np.random.random() < 0.9 else parent_1
        children.append(
            mutate(
                child,
                mutation_dist,
                hyperparam_mutation_dists,
                hyperparam_name_to_id_dict,
                search_space_limits,
            )
        )

    return children

def select_best_fits(population: List[List[float]], n_best: int, fitness_function: Callable[[List[float]], float]) -> List[List[float]]:
    return sorted(population, key=fitness_function)[:n_best]

def initialize_uniform_hypercube_dist(dim: int, dim_limits: List[Tuple[float,float]]) -> Callable[[], List[np.float64]]:
    assert len(dim_limits) == dim
    return lambda: [np.random.uniform(dim_limits[i][0], dim_limits[i][1]) for i in range(dim)]

def initialize_multivariate_normal_dist(mean: np.ndarray, sigma: np.ndarray) -> Callable[[], np.ndarray]:
    assert mean.shape[0] == sigma.shape[0]
    assert sigma.shape[0] == sigma.shape[1]
    return lambda: np.random.multivariate_normal(mean, sigma)

def initialize_lognormal_dist(mean: float, sigma: float) -> Callable[[], np.float64]:
    return lambda: np.random.lognormal(mean, sigma)

class DistributionMaker(Enum):
    INITIAL_UNIFORM_HYPERCUBE = initialize_uniform_hypercube_dist
    MULTIVARIATE_NORMAL = initialize_multivariate_normal_dist
    LOGNORMAL = initialize_lognormal_dist

@dataclass
class ESConfig:
    n_iterations: int = 100
    dim: int = 4
    search_space_limits: Optional[List[Tuple[float, float]]] = None
    mu: int = 100
    lamb: int = 300
    sigma_0: float = 1
    initial_pop_dist: Callable[[int, List[Tuple[float, float]]], Callable[[], List[float]]] = DistributionMaker.INITIAL_UNIFORM_HYPERCUBE
    opt_domain_mutation_dist: Optional[Callable[[np.ndarray, np.ndarray], Callable[[], np.ndarray]]] = None
    sigma_mutation_dist: Callable[[float, float], Callable[[], float]] = DistributionMaker.LOGNORMAL
    sigma_sigma: float = 0.1
    hyperparam_name_to_id_dict: Optional[Dict[str, int]] = None
    client_script_name: str = "client.py"
    opt_info_interval: int = 10

    def __post_init__(self):
        if not self.hyperparam_name_to_id_dict:
            self.hyperparam_name_to_id_dict = {"sigma": 0}
        if not self.search_space_limits:
            self.search_space_limits = [(-10, 10) for _ in range(self.dim)]
        self.initial_pop_dist_f = self.initial_pop_dist(self.dim, self.search_space_limits)
        if not self.opt_domain_mutation_dist:
            self.opt_domain_mutation_dist = lambda hyperparam_domain, hyperparam_name_to_id_dict: DistributionMaker.MULTIVARIATE_NORMAL(np.zeros(self.dim), np.eye(self.dim) * hyperparam_domain[hyperparam_name_to_id_dict["sigma"]])
        self.sigma_mutation_dist_f = self.sigma_mutation_dist(0, self.sigma_sigma)
def optimize(config: ESConfig) -> Tuple[float, List[float]]:
    population = initiate_population(config.mu, config.initial_pop_dist_f, [config.sigma_0])
    print(f"Iteration: {0} | Best Fitness: {compute_fitness(population[0], config.dim, config.client_script_name)}")
    for i in range(1, config.n_iterations+1):
        population = select_best_fits(
            create_children(
                population,
                config.lamb,
                config.opt_domain_mutation_dist,
                [config.sigma_mutation_dist_f],
                config.hyperparam_name_to_id_dict,
                config.search_space_limits,
            ),
            config.mu,
            lambda individual: compute_fitness(individual, config.dim, config.client_script_name),
        )
        if i % config.opt_info_interval == 0:
            print(f"Iteration: {i} | Best Fitness: {compute_fitness(population[0], config.dim, config.client_script_name)}")
    return compute_fitness(population[0]), population[0]


def _clip_to_bounds(vector: List[float], limits: List[Tuple[float, float]]) -> List[float]:
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(vector, limits)]


def _split_individual(individual: List[float], opt_dim: int) -> Tuple[List[float], List[float]]:
    return individual[:opt_dim], individual[opt_dim:]


def _adapt_jde_control_params(
    F: float,
    CR: float,
    tau_F: float,
    tau_CR: float,
    F_bounds: Tuple[float, float],
    CR_bounds: Tuple[float, float],
) -> Tuple[float, float]:
    if np.random.random() < tau_F:
        F = float(np.random.uniform(F_bounds[0], F_bounds[1]))
    if np.random.random() < tau_CR:
        CR = float(np.random.uniform(CR_bounds[0], CR_bounds[1]))
    return F, CR


def _jde_binomial_crossover(
    target: List[float],
    mutant: List[float],
    CR: float,
) -> List[float]:
    dim = len(target)
    j_rand = np.random.randint(dim)
    trial = [
        mutant[j] if np.random.random() < CR or j == j_rand else target[j]
        for j in range(dim)
    ]
    return trial


def _jde_rand1_mutant(
    x_r1: List[float],
    x_r2: List[float],
    x_r3: List[float],
    F: float,
) -> List[float]:
    return list(np.asarray(x_r1) + F * (np.asarray(x_r2) - np.asarray(x_r3)))


@dataclass
class JDEConfig:
    n_iterations: int = 100
    dim: int = 4
    search_space_limits: Optional[List[Tuple[float, float]]] = None
    population_size: int = 100
    F_init: float = 0.5
    CR_init: float = 0.9
    tau_F: float = 0.1
    tau_CR: float = 0.1
    F_bounds: Tuple[float, float] = (0.1, 1.0)
    CR_bounds: Tuple[float, float] = (0.0, 1.0)
    initial_pop_dist: Callable[[int, List[Tuple[float, float]]], Callable[[], List[float]]] = DistributionMaker.INITIAL_UNIFORM_HYPERCUBE
    hyperparam_name_to_id_dict: Optional[Dict[str, int]] = None
    client_script_name: str = "client.py"
    opt_info_interval: int = 10

    def __post_init__(self):
        if not self.hyperparam_name_to_id_dict:
            self.hyperparam_name_to_id_dict = {"F": 0, "CR": 1}
        if not self.search_space_limits:
            self.search_space_limits = [(-10, 10) for _ in range(self.dim)]
        self.initial_pop_dist_f = self.initial_pop_dist(self.dim, self.search_space_limits)


def optimize_jde(config: JDEConfig) -> Tuple[float, List[float]]:
    """jDE (Brest et al., 2006): DE/rand/1/bin with self-adaptive F and CR per individual."""
    fitness_fn = lambda individual: compute_fitness(
        individual, config.dim, config.client_script_name
    )
    population = initiate_population(
        config.population_size,
        config.initial_pop_dist_f,
        [config.F_init, config.CR_init],
    )
    best_idx = min(
        range(len(population)),
        key=lambda i: fitness_fn(population[i]),
    )
    print(
        f"Iteration: {0} | Best Fitness: {fitness_fn(population[best_idx])}"
    )

    for iteration in range(1, config.n_iterations + 1):
        for i in range(config.population_size):
            target, hyperparams = _split_individual(population[i], config.dim)
            F_id = config.hyperparam_name_to_id_dict["F"]
            CR_id = config.hyperparam_name_to_id_dict["CR"]
            F, CR = _adapt_jde_control_params(
                hyperparams[F_id],
                hyperparams[CR_id],
                config.tau_F,
                config.tau_CR,
                config.F_bounds,
                config.CR_bounds,
            )

            candidates = [j for j in range(config.population_size) if j != i]
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            x_r1, _ = _split_individual(population[r1], config.dim)
            x_r2, _ = _split_individual(population[r2], config.dim)
            x_r3, _ = _split_individual(population[r3], config.dim)
            mutant = _clip_to_bounds(
                _jde_rand1_mutant(x_r1, x_r2, x_r3, F),
                config.search_space_limits,
            )
            trial = _clip_to_bounds(
                _jde_binomial_crossover(target, mutant, CR),
                config.search_space_limits,
            )
            trial_individual = trial + [F, CR]

            if fitness_fn(trial_individual) <= fitness_fn(population[i]):
                population[i] = trial_individual

        if iteration % config.opt_info_interval == 0:
            best_idx = min(
                range(len(population)),
                key=lambda i: fitness_fn(population[i]),
            )
            print(
                f"Iteration: {iteration} | Best Fitness: {fitness_fn(population[best_idx])}"
            )

    best_idx = min(
        range(len(population)),
        key=lambda i: fitness_fn(population[i]),
    )
    return fitness_fn(population[best_idx]), population[best_idx]


def _search_space_bounds(
    limits: List[Tuple[float, float]],
) -> Tuple[List[float], List[float]]:
    lower = [lo for lo, _ in limits]
    upper = [hi for _, hi in limits]
    return lower, upper


def _search_space_center(limits: List[Tuple[float, float]]) -> List[float]:
    return [(lo + hi) / 2 for lo, hi in limits]


@dataclass
class CMAESConfig:
    dim: int = 4
    search_space_limits: Optional[List[Tuple[float, float]]] = None
    sigma_0: float = 0.3
    x0: Optional[List[float]] = None
    max_fevals: Optional[int] = 3000
    n_iterations: Optional[int] = None
    client_script_name: str = "client.py"
    opt_info_interval: int = 10
    seed: Optional[int] = None
    cma_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.search_space_limits:
            self.search_space_limits = [(-10, 10) for _ in range(self.dim)]
        if self.x0 is not None and len(self.x0) != self.dim:
            raise ValueError(f"x0 must have length {self.dim}, got {len(self.x0)}")


@dataclass
class PSOMatlabConfig:
    matlab_script_name: str = "main"
    matlab_source_dir: str = "src/matlab"
    matlab_command: str = "matlab"
    matlab_flags: Optional[List[str]] = None
    working_directory: str = "."
    use_shared_engine: bool = True
    shared_engine_name: str = "ubuntu_matlab"
    max_iter: int = 50
    particle_number: int = 100
    opt_info_interval: int = 10
    c1: float = 2.05
    c2: float = 2.05
    rang_coef: float = 0.6
    random_seed: Optional[int] = None
    echo_logs: bool = True

    def __post_init__(self):
        if self.matlab_flags is None:
            self.matlab_flags = ["-nodesktop", "-nosplash", "-nojvm", "-batch"]

    def matlab_seed_arg(self) -> float:
        """MATLAB main() uses a negative value to mean 'do not fix the RNG seed'."""
        if self.random_seed is None:
            return -1.0
        return float(self.random_seed)


def optimize_cma_es(config: CMAESConfig) -> Tuple[float, List[float]]:
    """CMA-ES via pycma with box constraints on the optimization domain."""
    if cma is None:
        raise ImportError("CMA-ES requires package 'cma'. Install it with: pip install cma")
    fitness_fn = lambda individual: compute_fitness(
        individual, config.dim, config.client_script_name
    )
    lower, upper = _search_space_bounds(config.search_space_limits)
    x0 = config.x0 or _search_space_center(config.search_space_limits)

    opts: Dict[str, Any] = {
        "bounds": [lower, upper],
        "verbose": -9,
    }
    if config.max_fevals is not None:
        opts["maxfevals"] = config.max_fevals
    if config.n_iterations is not None:
        opts["maxiter"] = config.n_iterations
    if config.seed is not None:
        opts["seed"] = config.seed
    opts.update(config.cma_options)

    es = cma.CMAEvolutionStrategy(x0, config.sigma_0, opts)
    generation = 0

    while not es.stop():
        solutions = es.ask()
        fitnesses = [fitness_fn(list(solution)) for solution in solutions]
        es.tell(solutions, fitnesses)
        if generation % config.opt_info_interval == 0:
            print(
                f"Generation: {generation} | Best Fitness: {es.result.fbest} | "
                f"Evaluations: {es.result.evaluations}"
            )
        generation += 1

    best_solution = list(es.result.xbest)
    return float(es.result.fbest), best_solution


def _parse_pso_best_fitness(log_text: str) -> Optional[float]:
    final_match = re.search(
        r"PSO_FINAL_FITNESS:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        log_text,
    )
    if final_match:
        return float(final_match.group(1))
    fitness_matches = re.findall(
        r"Fitness\(best\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        log_text,
    )
    if fitness_matches:
        return float(fitness_matches[-1])
    return None


def optimize_pso_matlab(config: PSOMatlabConfig) -> Tuple[float, str]:
    """
    Run MATLAB PSO optimizer from src/matlab and return best fitness and full stdout.

    Best fitness is taken from the function return value when available, otherwise
    parsed from ``PSO_FINAL_FITNESS`` or ``Fitness(best):`` log lines.
    """
    source_dir = str((Path(config.working_directory) / config.matlab_source_dir).resolve())
    captured_stdout = ""
    best_from_engine: Optional[float] = None

    class _TeeWriter(io.StringIO):
        def __init__(self, stream):
            super().__init__()
            self._stream = stream

        def write(self, s):
            if config.echo_logs and self._stream is not None:
                self._stream.write(s)
                self._stream.flush()
            return super().write(s)

    if config.use_shared_engine:
        if matlab is None:
            raise ImportError(
                "Shared MATLAB engine mode requires package 'matlabengine'. "
                "Install it or set use_shared_engine=False."
            )
        try:
            eng = matlab.engine.connect_matlab(config.shared_engine_name)
            eng.addpath(source_dir, nargout=0)
            out_tee = _TeeWriter(sys.stdout)
            err_tee = _TeeWriter(sys.stderr)
            eng.cd(source_dir, nargout=0)
            matlab_args = (
                int(config.max_iter),
                int(config.particle_number),
                int(config.opt_info_interval),
                float(config.c1),
                float(config.c2),
                float(config.rang_coef),
                config.matlab_seed_arg(),
            )
            try:
                result = eng.feval(
                    config.matlab_script_name,
                    *matlab_args,
                    nargout=1,
                    stdout=out_tee,
                    stderr=err_tee,
                )
                best_from_engine = float(result)
            except Exception:
                eng.feval(
                    config.matlab_script_name,
                    *matlab_args,
                    nargout=0,
                    stdout=out_tee,
                    stderr=err_tee,
                )
            stdout = out_tee.getvalue()
            captured_stdout = stdout
        except Exception as err:
            raise RuntimeError(
                "MATLAB PSO execution via shared engine failed. "
                f"engine={config.shared_engine_name!r} | expr={config.matlab_script_name!r} | error={err}"
            ) from err
        stderr = ""
        returncode = 0
    else:
        flags = config.matlab_flags or ["-nodesktop", "-nosplash", "-nojvm", "-batch"]
        batch_expr = (
            f"cd('{source_dir}'); "
            f"{config.matlab_script_name}({int(config.max_iter)},"
            f"{int(config.particle_number)},{int(config.opt_info_interval)},"
            f"{float(config.c1)},{float(config.c2)},{float(config.rang_coef)},"
            f"{config.matlab_seed_arg()})"
        )
        cmd = [config.matlab_command, *flags, batch_expr]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=config.working_directory,
        )
        stdout = result.stdout or ""
        captured_stdout = stdout
        stderr = result.stderr or ""
        returncode = result.returncode
        if config.echo_logs and stdout:
            print(stdout, end="")

    if returncode != 0:
        raise RuntimeError(
            "MATLAB PSO execution failed. "
            f"returncode={returncode} | stdout={stdout!r} | stderr={stderr!r}"
        )

    best_fitness = best_from_engine
    if best_fitness is None:
        best_fitness = _parse_pso_best_fitness(captured_stdout)
    if best_fitness is None:
        raise RuntimeError(
            "MATLAB PSO finished but best fitness could not be determined. "
            "Expected function output, 'PSO_FINAL_FITNESS', or 'Fitness(best)' in logs. "
            f"stdout={stdout!r} | stderr={stderr!r}"
        )
    return float(best_fitness), captured_stdout
