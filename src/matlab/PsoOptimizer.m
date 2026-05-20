classdef PsoOptimizer < handle
  properties
    params_
    dimension_
    bounds_
    Kap_
    population_
    best_fitness_mat_
    best_population_
    best_point_
    xmin_
    xmax_
    xmax_value_ = 4
  end
  methods
    function obj = PsoOptimizer(pso_params, dimension, bounds)
      obj.params_ = pso_params;
      obj.dimension_ = dimension;
      obj.bounds_ = bounds;

      obj.xmin_ = repmat(-obj.xmax_value_, obj.dimension_, 1);
      obj.xmax_ = repmat(obj.xmax_value_, obj.dimension_, 1);

      phi = pso_params.c1 + pso_params.c2;
      obj.Kap_ = 2 / abs(2 - phi - sqrt(phi^2 - 4 * phi));
      obj.best_point_ = 1;
      obj.best_fitness_mat_ = zeros(obj.params_.max_iter, 1);

      pop_center = mean(obj.bounds_.space_range, 2);
      pop_range = range(obj.bounds_.space_range, 2) * obj.params_.rang_coef;

      % Initialization of the swarm
      % (:,1,:) Position
      % (:,2,:) Velocity
      % (:,3,:) pbest
      % (:,4,:) pbest fitness
      obj.population_ = zeros(obj.params_.particle_number, 4, obj.dimension_);

      % Compute the box center and a shrunken span to initialize positions
      for index = 1:obj.params_.particle_number
          % Initialization of particles positions
          obj.population_(index, 1, :) = pop_center + (pop_range.* (rand(obj.dimension_,1) - 0.5));  
          % Initialization of particles velocities
          obj.population_(index, 2, :) = obj.bounds_.vel_clamp.* (rand(obj.dimension_, 1) - 0.5) * 2;   
      end
      obj.population_(:,4,:) = 1e40;
    end

    function update_best(obj, iter_num, fitness_values) 
      for n = 1:obj.params_.particle_number
        if fitness_values(n) < obj.population_(n, 4, 1)
          obj.population_(n, 3, :) = obj.population_(n, 1, :);
          obj.population_(n, 4, 1) = fitness_values(n);
        end
      end

      [~, gbest] = min(obj.population_(:, 4, 1));
      obj.best_point_ = gbest;
      gbest_val = obj.population_(obj.best_point_, 4, 1);
      disp(['Iteration: ' num2str(iter_num) '  Fitness: ' num2str(fitness_values(obj.best_point_)) ' Fitness(best): ' num2str(gbest_val)]);
    end

    function update_parameters(obj) 
      for n = 1:obj.params_.particle_number
          for d = 1:obj.dimension_
              r1 = rand;
              r2 = rand;
              x = obj.population_(n, 1, d);
              v = obj.population_(n, 2, d);
              p = obj.population_(n, 3, d); 
              g = obj.population_(obj.best_point_, 3, d); 
              v = obj.Kap_ * (v + obj.params_.c1 * r1 * (p - x) + obj.params_.c2 * r2* (g - x));
              v = min(max(v, -obj.bounds_.vel_clamp(d)), obj.bounds_.vel_clamp(d));
              x = x + v;
              % Absorbing walls: When a particle hits the boundary of the
              % solution space, the velocity is zeroed in that dimension.
              if x < obj.xmin_(d)
                  x = obj.xmin_(d);
                  v = 0; % absorb: kill velocity in this dim
              elseif x > obj.xmax_(d)
                  x = obj.xmax_(d);
                  v = 0; % absorb: kill velocity in this dim
              end
              % % Update
              obj.population_(n, 2, d) = v;
              obj.population_(n, 1, d) = x;
          end
      end
    end

    function [best_population_, best_fitness_mat_] = optimize(obj, evaluator)
      fitness_values = inf(1, obj.params_.particle_number);
      rng(1, 'twister');
      for iter_cnt = 1:obj.params_.max_iter
          parfor particle_cnt = 1:obj.params_.particle_number
              try
                  working_point = obj.population_(particle_cnt, :, :);
                  working_point = squeeze(working_point(1, 1, :));
                  fitness_wp = evaluator.evaluate(working_point); 
		  fitness_values(particle_cnt) = fitness_wp;
              catch ME
		  disp(ME);
                  fitness_values(particle_cnt) = 1e6;
                  disp(['Evaluation for particle no. ' num2str(particle_cnt) ' was aborted']);
              end
          end
          obj.update_best(iter_cnt, fitness_values);
          obj.update_parameters();
      end
    end
  end
end
