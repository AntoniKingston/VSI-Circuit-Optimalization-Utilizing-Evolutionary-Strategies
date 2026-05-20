function [vel_clamp, space_range, dimension] = make_init_pso_bounds(sim_params, xi_mode_flag, xi_max_value, xi_max_velocity)
  xi_blk = [];
  xi_dim = 0;
  xi_vel_clamp = [];
  if xi_mode_flag == 1
    xi_blk = [0 xi_max_value];
    xi_dim = 1;
    xi_vel_clamp = xi_max_velocity;
  end
  if sim_params.R_mode == 1
    R_blk = [-sim_params.Rmax sim_params.Rmax];
    R_blks = repmat(R_blk, sim_params.nR, 1);
    R_vel_clamp = repmat(sim_params.Rmax_vel, sim_params.nR, 1);
  else
    nR = 0;
    R_blks = [];
    R_vel_clamp = [];
  Q_blk = [-sim_params.Qmax sim_params.Qmax];
  Q_blks = repmat(Q_blk, sim_params.nQ, 1);
  Q_vel_clamp = repmat(sim_params.Qmax_vel, sim_params.nQ, 1);

  vel_clamp = [Q_vel_clamp; R_vel_clamp; xi_vel_clamp];
  space_range = [Q_blks; R_blks; xi_blk];
  dimension = sim_params.nQ + (sim_params.R_mode==1)*sim_params.nR + xi_dim;
end
