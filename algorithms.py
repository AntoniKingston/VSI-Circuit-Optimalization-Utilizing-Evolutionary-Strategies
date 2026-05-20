from typing import List, Callable
import subprocess
from dataclasses import dataclass
import numpy as np

# Function works only if server is already running
def compute_fitness(individual: List[float], opt_domain_pos_len: int = 4, client_script_name: str = "client.py") -> float:
    individual = individual[:opt_domain_pos_len]
    data_str = " ".join(map(str, individual), )
    cmd = f"python {client_script_name} -d {data_str}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out_str = result.stdout
    return float(out_str.split(" ")[4])

def initiate_population(n_individuals: int, initial_dist: Callable[[], List[float]], initial_hyperparam_values: List[float]) -> list[list[float]]:
    population = []
    for _ in range(n_individuals):
        individual = initial_dist()
        individual += initial_hyperparam_values
        population.append(individual)
    return population

def crossover(parent_1: list[float], parent_2: list[float], crossover_point: int) -> list[float]:
    assert len(parent_1) == len(parent_2)
    assert crossover_point < len(parent_1)
    child = parent_1[:crossover_point] + parent_2[crossover_point:]
    return child

def mutate(individual: list[float], mutation_dist: Callable[[], list[float]], hyperparam_mutation_dists: List[Callable[[], float]]) -> list[float]:
    opt_domain_pos_len = len(individual) - len(hyperparam_mutation_dists)
    opt_domain = individual[:opt_domain_pos_len]
    hyperparam_domain = individual[opt_domain_pos_len:]
    opt_domain += mutation_dist()
    for i, hyperparam_mutation_dist in enumerate(hyperparam_mutation_dists):
        hyperparam_domain[i] = hyperparam_mutation_dist()
    return opt_domain + hyperparam_domain



@dataclass
class ESConfig:
    n_iterations: int = 100
    n_dim: int = 4
    mu: int = 100
    lamb: int = 300
    sigma_0: float = 0.1
    pos_mutation_dist: Callable[[int, float], list[float]] = lambda n_dim, sigma: np.random.normal(0, sigma, n_dim)
    sigma_sigma: float = 0.1
    sigma_mutation_dist: Callable[[float], float] = lambda sigma_sigma: np.random.lognormal(0, sigma_sigma)

class ES:
    def __init__(self, config: ESConfig) -> None:
        self.config = config


