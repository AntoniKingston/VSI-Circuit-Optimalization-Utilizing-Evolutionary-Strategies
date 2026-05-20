classdef DLQREvaluator
  properties
    simulator_
  end
  methods
    function obj =  DLQREvaluator(simulator)
      obj.simulator_ = simulator;
    end
    function [fitness] = evaluate(obj, working_point)
      if ~iscolumn(working_point)
      	working_point = working_point';
      end
      [K, Kx, Kr, Ku] = calculate_k_matrix(working_point, obj.simulator_.params_);
      Acl = obj.simulator_.params_.Ad_aug - obj.simulator_.params_.Bd_aug * K;
      spec = max(abs(eig(Acl)));
      if spec >= 1
          fitness = 1e6;
          disp('One of the particles is too close of the instable area!')
          return;
      end

      try
          fitness = obj.simulator_.run(Kx, Kr, Ku);
      catch ME
          fprintf('\nSimulation failed: %s\n', ME.message)
          disp(getReport(ME, 'extended'))
          fitness = 1e6;
          return;
      end
    end
  end
end
