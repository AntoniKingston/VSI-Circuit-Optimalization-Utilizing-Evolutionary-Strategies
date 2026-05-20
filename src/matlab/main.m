clc;
echo off;

sim_params = SimulatorParameters(1e-3, 22e-6, 35e-3, 20e3, 50, 700, 0.3, 100, 40e-3, 3e-5);
simulator = Simulator(sim_params);
evaluator = DLQREvaluator(simulator);

pso_params = PsoOptimizerParameters();
[vel_clamp, space_range, dimension] = make_init_pso_bounds(sim_params, 0, 0, 0);
search_bounds = PsoOptimizerSearchBounds(vel_clamp, space_range);
pso_optimizer = PsoOptimizer(pso_params, dimension, search_bounds);
pso_optimizer.optimize(evaluator);
