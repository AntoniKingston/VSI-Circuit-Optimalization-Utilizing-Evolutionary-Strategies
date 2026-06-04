function best_fitness = main(max_iter, particle_number, info_interval, c1, c2, rang_coef, random_seed)
% PSO entry point for Python experiments. Returns scalar best_fitness.
% random_seed < 0 leaves MATLAB's RNG unchanged.

if nargin < 1
    max_iter = 50;
end
if nargin < 2
    particle_number = 100;
end
if nargin < 3
    info_interval = 10;
end

clc;
echo off;

if nargin >= 7 && ~isempty(random_seed) && random_seed >= 0
    rng(random_seed);
end

sim_params = SimulatorParameters(1e-3, 22e-6, 35e-3, 20e3, 50, 700, 0.3, 100, 40e-3, 3e-5);
simulator = Simulator(sim_params);
evaluator = DLQREvaluator(simulator);

pso_params = PsoOptimizerParameters();
pso_params.max_iter = max_iter;
pso_params.particle_number = particle_number;
pso_params.info_interval = info_interval;
if nargin >= 4 && ~isempty(c1)
    pso_params.c1 = c1;
end
if nargin >= 5 && ~isempty(c2)
    pso_params.c2 = c2;
end
if nargin >= 6 && ~isempty(rang_coef)
    pso_params.rang_coef = rang_coef;
end

[vel_clamp, space_range, dimension] = make_init_pso_bounds(sim_params, 0, 0, 0);
search_bounds = PsoOptimizerSearchBounds(vel_clamp, space_range);
pso_optimizer = PsoOptimizer(pso_params, dimension, search_bounds);
[best_fitness, best_solution] = pso_optimizer.optimize(evaluator);
fprintf('PSO_FINAL_FITNESS: %.17g\n', best_fitness);
fprintf('PSO_BEST_SOLUTION: %s\n', mat2str(best_solution, 6));
end
