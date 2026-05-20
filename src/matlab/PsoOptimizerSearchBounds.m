classdef PsoOptimizerSearchBounds
  properties 
    vel_clamp
    space_range
  end
  methods
    function obj = PsoOptimizerSearchBounds(vc, sr)
      obj.vel_clamp = vc;
      obj.space_range = sr;
    end
  end
end
