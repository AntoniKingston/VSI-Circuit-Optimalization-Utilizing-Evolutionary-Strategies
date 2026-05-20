from typing import List, Callable, Tuple, Optional, Dict
import subprocess
from dataclasses import dataclass
import numpy as np
from enum import Enum

# Function works only if server is already running
def compute_fitness(individual: List[float], opt_domain_pos_len: int = 4, client_script_name: str = "client.py") -> float:
    individual = individual[:opt_domain_pos_len]
    data_str = " ".join(map(str, individual), )
    cmd = f"python {client_script_name} -d {data_str}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out_str = result.stdout
    return float(out_str.split(" ")[4])

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

def mutate(individual: List[float], mutation_dist: Callable[[List, Dict[str, int]], Callable[[], List[float]]], hyperparam_mutation_dists: List[Callable[[], float]], hyperparam_name_to_id_dict: Dict[str, int]) -> List[float]:
    opt_domain_pos_len = len(individual) - len(hyperparam_mutation_dists)
    opt_domain = individual[:opt_domain_pos_len]
    hyperparam_domain = individual[opt_domain_pos_len:]
    mutation_dist_f = mutation_dist(opt_domain, hyperparam_name_to_id_dict)
    opt_domain += mutation_dist_f()
    for i, hyperparam_mutation_dist in enumerate(hyperparam_mutation_dists):
        hyperparam_domain[i] += float(hyperparam_mutation_dist())
    return list(opt_domain) + list(hyperparam_domain)

# We allow for no crossover (crossover between identical parents)
def create_children(parents: List[List[float]], n_children: int, mutation_dist: Callable[[List, Dict[str, int]], Callable[[], List[float]]], hyperparam_mutation_dists: List[Callable[[], float]], hyperparam_name_to_id_dict: Dict[str, int]) -> List[List[float]]:
    children = []
    for _ in range(n_children):
        id_1, id_2 = np.random.choice(len(parents), 2, replace=False)
        parent_1 = parents[id_1]
        parent_2 = parents[id_2]
        child = crossover(parent_1, parent_2) if np.random.random() < 0.9 else parent_1
        children.append(mutate(child, mutation_dist, hyperparam_mutation_dists, hyperparam_name_to_id_dict))

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
            self.search_space_limits = [(-1, 1) for _ in range(self.dim)]
        self.initial_pop_dist_f = self.initial_pop_dist(self.dim, self.search_space_limits)
        if not self.opt_domain_mutation_dist:
            self.opt_domain_mutation_dist = lambda hyperparam_domain, hyperparam_name_to_id_dict: DistributionMaker.MULTIVARIATE_NORMAL(np.zeros(self.dim), np.eye(self.dim) * hyperparam_domain[hyperparam_name_to_id_dict["sigma"]])
        self.sigma_mutation_dist_f = self.sigma_mutation_dist(0, self.sigma_sigma)
def optimize(config: ESConfig) -> Tuple[float, List[float]]:
    population = initiate_population(config.mu, config.initial_pop_dist_f, [config.sigma_0])
    print(f"Iteration: {0} | Best Fitness: {compute_fitness(population[0], config.dim, config.client_script_name)}")
    for i in range(1, config.n_iterations+1):
        population = select_best_fits(create_children(population, config.lamb, config.opt_domain_mutation_dist, [config.sigma_mutation_dist_f], config.hyperparam_name_to_id_dict), config.mu, lambda individual:compute_fitness(individual, config.dim, config.client_script_name))
        if i % config.opt_info_interval == 0:
            print(f"Iteration: {i} | Best Fitness: {compute_fitness(population[0], config.dim, config.client_script_name)}")
    return compute_fitness(population[0]), population[0]


