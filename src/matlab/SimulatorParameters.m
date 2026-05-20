classdef SimulatorParameters
  properties
    Lf
    Cf
    Rf
    fsw
    f
    V_ref
    Tsim
    Ts
    Vdc
    w
    w_vec
    n_h
    A_ref
    t_settling
    beta_c
    Ad
    Bd
    Ard
    Brd
    Ad_aug
    Bd_aug
    Ha
    nQ
    Qmax 
    Qmax_vel
    nR
    Rmax
    Rmax_vel 
    R_mode = 0
  end
  methods
    function obj = SimulatorParameters(Lf, Cf, Rf, fsw, f, Vdc, Tsim, V_ref, t_settling, beta_c)
      obj.Lf = Lf;
      obj.Cf = Cf;
      obj.Rf = Rf;
      A = [-obj.Rf/obj.Lf  0     -1/obj.Lf   0    ;
            0     -obj.Rf/obj.Lf  0     -1/obj.Lf ;
            1/obj.Cf   0      0      0    ;
            0      1/obj.Cf   0      0   ];
      B = [1/obj.Lf   0    ;
            0     1/obj.Lf ;
            0     0    ;
            0     0   ];
      P = [ 0     0    ;
            0     0    ;
           -1/obj.Cf  0    ;
            0    -1/obj.Cf];

      nx = size(B, 1);
      nu = size(B, 2);

      C = eye(nx);
      D = zeros(nx, nu);

      ss_VSI = ss(A, B, C, D);

      obj.fsw = fsw;
      obj.Ts = 1 / (2 * obj.fsw);
      [Ad, Bd, ~, ~] = ssdata(c2d(ss_VSI, obj.Ts, 'zoh'));

      ss_VSI = ss(A, B, eye(nx), []);
      [Ad, Bd, ~, ~] = ssdata(c2d(ss_VSI, obj.Ts, 'zoh'));
      obj.Ad = Ad;
      obj.Bd = Bd;

      Ad_delay = [obj.Ad obj.Bd; zeros(nu,nx) zeros(nu)];
      Bd_delay = [zeros(nx,nu); eye(nu)];

      obj.f = f;
      obj.w = 2 * pi * obj.f;
      Ar1 = [0 obj.w; -obj.w 0];
      Br1 = [0; 1];
      [Ard1, Brd1, ~, ~] = ssdata(c2d(ss(Ar1, Br1, eye(2), []), obj.Ts, 'tustin'));
      Ard1 = blkdiag(Ard1, Ard1);
      Brd1 = blkdiag(Brd1, Brd1);

      obj.w_vec = obj.w*[1];
      obj.n_h = length(obj.w_vec);

      obj.Ard = blkdiag(Ard1);

      obj.Brd = [Brd1];
      nxr = size(obj.Brd, 1);
      nur = size(obj.Brd, 2);

      C_delay = [C zeros(nx, nu)];

      obj.Ha = [0 0 1 0 ; 0 0 0 1];

      Hx = C_delay(3:4,:);

      obj.Ad_aug = [Ad_delay  zeros((nx + nu), nxr) ; -obj.Brd*Hx   obj.Ard];
      obj.Bd_aug = [Bd_delay; zeros(nxr, nu)];

      obj.nQ = nx  / 2 + nu / 2 + obj.n_h;
      obj.Qmax = 4;
      obj.Qmax_vel = 1;

      obj.nR = size(obj.Bd_aug, 2) / 2;
      obj.Rmax = 2;
      obj.Rmax_vel = 0.1;

      obj.Vdc = Vdc;
      obj.Tsim = Tsim;
      obj.V_ref = V_ref;
      obj.t_settling = t_settling;
      obj.A_ref = obj.V_ref;
      obj.beta_c = beta_c;
    end
  end
end
