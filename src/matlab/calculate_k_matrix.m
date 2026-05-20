function [K, Kx, Ku, Kr] = calculate_k_matrix(working_point, sim_params) 
  nx  = size(sim_params.Bd, 1); 
  nu  = size(sim_params.Bd, 2);
  nxr = size(sim_params.Brd, 1);

  states_per_h = nxr / sim_params.n_h;

  idx_x_end = nx / 2;
  idx_u_end = idx_x_end + nu / 2;

  working_point_x  = working_point(1:idx_x_end);
  working_point_u  = working_point(idx_x_end+1:idx_u_end);
  working_point_hr = working_point(idx_u_end+1:idx_u_end+sim_params.n_h);

  Qx = repelem(working_point_x, 2);
  Qu = repelem(working_point_u, 2);
  Qr_vec = zeros(nxr, 1);

  offset = 0;
  for i = 1:sim_params.n_h
      qi  = 10.^working_point_hr(i);   % optimization variable in log10 scale
      wi2 = sim_params.w_vec(i)^2;
      Qr_block = qi * wi2 * ones(states_per_h,1);  % same for all states of this harmonic
      Qr_vec(offset+1:offset+states_per_h) = Qr_block;
      offset = offset + states_per_h;
  end

  if isrow(Qu)
      Qu = Qu';
  end

  Q_diag = [10.^Qx; 10.^Qu; Qr_vec];
  Q = diag(Q_diag);

  if sim_params.R_mode == 1
      Ru = repelem(working_point(end - nu/2 + 1:end), 2);
      R = diag((10.^Ru)');
  else
      R = diag(repelem(1, nu));
  end

  if any(R(:) < 0,1) || any(Q(:) < 0,1)
      disp('One of the elements of the Q and R matrixes is negative!')
      fitness = 1e6; 
      return;
  end
  
  try
      [K, ~, ~] = dlqr(sim_params.Ad_aug, sim_params.Bd_aug, Q, R);
  catch ME
      disp('Error in the LQR')
      disp(ME);
      fitness = 1e6; 
      return;
  end

  Kx = K(:, 1:nx);
  Ku = K(:, (nx + 1):(nx + nu));
  Kr = K(:, (nx + nu + 1):end);
end
