function main(max_iter, particle_number, info_interval)
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

sim_params = SimulatorParameters(1e-3, 22e-6, 35e-3, 20e3, 50, 700, 0.3, 100, 40e-3, 3e-5);
simulator = Simulator(sim_params);
evaluator = DLQREvaluator(simulator);

pso_params = PsoOptimizerParameters();
pso_params.max_iter = max_iter;
pso_params.particle_number = particle_number;
pso_params.info_interval = info_interval;

[vel_clamp, space_range, dimension] = make_init_pso_bounds(sim_params, 0, 0, 0);
search_bounds = PsoOptimizerSearchBounds(vel_clamp, space_range);
pso_optimizer = PsoOptimizer(pso_params, dimension, search_bounds);
pso_optimizer.optimize(evaluator);
end
