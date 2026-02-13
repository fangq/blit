function [x, flag, relres, iter, resv] = blqmr(A, B, qtol, maxit, M1, M2, x0, varargin)
%
% [x,flag,relres,iter,resv] = blqmr(A,B,qtol,maxit,M1,M2,x0,opt)
%
%   Block Quasi-Minimal-Residual (BL-QMR) iterative solver for sparse
%   complex symmetric systems (suited for FEM modeling) in the form of
%                  A*[x1, x2, ...] = [b1, b2, ...]
%
% Author: Qianqian Fang <q.fang at neu.edu>
%
% Address: 360 Huntington Ave, ISEC 206
%          Boston, MA 02115, USA
%          Dept. of Bioengineering, Northeastern University, USA
%
% Input:
%      A: a full/sparse symmetric NxN matrix, can be either real or complex
%      B: an N x Nrhs array, representing Nrhs right-hand-side vectors
%      qtol: quasi-residual convergence tolerance, a number in [0,1]; if
%            qtol=[] or ignored, blqmr sets qtol=1e-6
%      maxit: maximum iteration number, if maxit=[] or ignored, blqmr uses
%            min(N,20)
%      M1, M2: preconditioning matrix M=M1*M2. If only M1 is given, M=M1;
%            if both M1 and M2 are empty, preconditioning is disabled. With
%            M, blqmr solves for the precond. system inv(M)*Ax=inv(M)*B;
%            the inversion is applied by (M1*M2)\(B-Ax) -> (M2\(M1\(B-Ax)))
%            M is the left preconditioning matrix where inv(M)*A approx. I
%      x0: the initial guess of x, dimension N x Nrhs; if x0=[] or ignored,
%            blqmr sets x0 to zeros(size(B))
%      opt: an optional input for additional parameters:
%            opt.residual: if 1, blqmr uses the true residual B-A*x instead
%                of the quasi-residual
%            opt.precond: automatic preconditioner creation, can be:
%                'ilu'   - ILU(0) factorization (real matrices only)
%                'ilutp' - ILUTP with drop tolerance (real matrices only)
%                'ichol' - Incomplete Cholesky (SPD matrices)
%                'diag'  - Diagonal (Jacobi) preconditioner
%            opt.droptol: drop tolerance for ILUTP (default: 1e-3)
%            opt.blocksize: solve RHS in batches of this size (default: all
%                RHS at once). When blocksize=1, solves one RHS at a time
%                (more robust for complex symmetric systems). Useful for
%                memory management or avoiding block algorithm instabilities.
%            opt.usefortran: if set to 0, force using the MATLAB native
%                solver even if the Fortran MEX blqmr_ is available
%                (default: 1 - use MEX when available)
%
% Output:
%      x: the solution, dimension N x Nrhs
%      flag: if flag=0, blqmr converges below qtol within maxit iterations;
%            if flag=1, blqmr fails to converge below qtol within maxit;
%            if flag=2, M1 and/or M2 are rank-deficient
%            if flag=3, blqmr stalls, residual no longer reduce
%      relres: the relative residual of the last iteration
%      iter: the last iteration number
%      resv: the vector of relective residuals for all iterations
%
% Examples:
%
%      n=100;
%      di=ones(n,1);
%      A=spdiags([-di 2*di -di],-1:1,n,n);
%      b=rand(n,4);
%      x = blqmr(A,b,[],100);
%
%      [x2,flag,res,iter,resv]=blqmr(A,b,1e-5,1000);
%      semilogy(resv);
%
%      A=A+i*A/2;
%      R=cholinc(A,0.8);
%      [x3,flag,res,iter,resv]=blqmr(A,b,1e-5,1000,R,R');
%
% Reference:
%      Fang Q, Meaney PM, Geimer SD, Streltsov AV, Paulsen KD, "Microwave image
%      reconstruction from 3D fields coupled to 2D parameter estimation," IEEE
%      Transactions on Medical Imaging, vol. 23, pp. 475-484, Apr. 2004.
%
%      Boyse, W. E., Seidl, A. A. (1996) A block QMR method for computing
%      multiple simultaneous solutions to complex symmetric systems.
%      SIAM J. Sci. Comput. 17, 263-274.
%
%  -- this file is part of Blit sparse solver library
%     URL: http://blit.sf.net
%
%  License:
%      BSD or LGPL or GPL, see License.txt for more details
%

%
% k-index range for all variables:
%      v:101,vt:1,beta:01,alpha:0,omega101,theta:0,Qa:10,Qb:210
%      Qc:10,Qd:210,zeta:0,zetat:0,eta:0,taot:01,tau:0,p:210
%

%% initialization of BLQMR

[n, m] = size(B);

% Parse opt structure first
opt = [];
if (nargin > 7 && ~isempty(varargin{:}))
    opt = varargin{1};
    if (~isstruct(opt))
        error('opt must be a structure');
    end
end

% Check for blocksize option - solve RHS in batches for robustness/memory
blocksize = m;  % default: solve all RHS at once
if ~isempty(opt) && isfield(opt, 'blocksize')
    blocksize = opt.blocksize;
    if blocksize < 1
        blocksize = 1;
    end
end

% If blocksize < m, solve in batches
if blocksize < m
    x = zeros(n, m);
    if ~isreal(A)
        x = complex(x);
    end

    % Calculate number of batches
    nbatches = ceil(m / blocksize);
    flags = zeros(1, nbatches);
    iters = zeros(1, nbatches);
    relress = zeros(1, nbatches);
    resv = [];

    % Remove blocksize from opt to avoid infinite recursion
    opt_batch = opt;
    opt_batch.blocksize = [];

    for batch = 1:nbatches
        % Calculate column indices for this batch
        col_start = (batch - 1) * blocksize + 1;
        col_end = min(batch * blocksize, m);
        cols = col_start:col_end;

        % Extract batch data
        B_batch = B(:, cols);
        x0_batch = [];
        if exist('x0', 'var') && ~isempty(x0)
            x0_batch = x0(:, cols);
        end

        % Solve this batch
        [x(:, cols), flags(batch), relress(batch), iters(batch), ~] = ...
            blqmr(A, B_batch, qtol, maxit, M1, M2, x0_batch, opt_batch);
    end

    flag = max(flags);    % Worst case flag
    relres = max(relress); % Worst case residual
    iter = max(iters);    % Max iterations
    return
end

%% Try Fortran MEX acceleration (blqmr_)
%  Conditions: no custom M1/M2 provided, sparse A, no function handle precond,
%  and user did not disable it via opt.usefortran=0

use_fortran = exist('blqmr_', 'file') == 3;  % 3 = MEX file

% Check if user disabled Fortran
if use_fortran && ~isempty(opt) && isfield(opt, 'usefortran')
    use_fortran = opt.usefortran;
end

% Cannot use Fortran MEX when custom M1/M2 matrices are provided
if use_fortran && ((nargin >= 5 && ~isempty(M1)) || (nargin >= 6 && ~isempty(M2)))
    % User supplied custom preconditioner - fall through to MATLAB solver
    % unless opt.precond was set (which we can map to Fortran pcond_type)
    if isempty(opt) || ~isfield(opt, 'precond')
        use_fortran = false;
    end
end

% Cannot use Fortran MEX with function handle preconditioners
if use_fortran && nargin >= 5 && ~isempty(M1) && isa(M1, 'function_handle')
    use_fortran = false;
end

% Fortran MEX requires sparse input
if use_fortran && ~issparse(A)
    use_fortran = false;
end

% opt.residual (true residual mode) is not supported by Fortran backend
if use_fortran && ~isempty(opt) && isfield(opt, 'residual') && opt.residual
    use_fortran = false;
end

if use_fortran
    % Map parameters for Fortran MEX
    if nargin < 3 || isempty(qtol), qtol = 1e-6; end
    if nargin < 4 || isempty(maxit), maxit = min(n, 20); end

    % Map preconditioner type to Fortran integer code
    %   0 = none, 1 = ILU-left, 2 = ILU-split, 3 = Jacobi-split
    pcond_type = 1;  % default: ILU-left
    droptol_f = 0.001;

    if ~isempty(opt) && isfield(opt, 'precond')
        ptype = opt.precond;
        if ischar(ptype) || isstring(ptype)
            switch lower(char(ptype))
                case {'ilu', 'ilutp', 'ilut', 'ilu0'}
                    pcond_type = 2;  % ILU-split (UMFPACK-based)
                case {'diag', 'jacobi'}
                    pcond_type = 3;  % Jacobi-split
                case 'none'
                    pcond_type = 0;
                otherwise
                    pcond_type = 1;  % default ILU-left
            end
        elseif isnumeric(ptype)
            pcond_type = ptype;
        end
    elseif (nargin < 5 || isempty(M1)) && (nargin < 6 || isempty(M2))
        % No preconditioner requested at all
        pcond_type = 0;
    end

    if ~isempty(opt) && isfield(opt, 'droptol')
        droptol_f = opt.droptol;
    end

    % Map blocksize to nblock for OpenMP
    nblock = 0;
    if ~isempty(opt) && isfield(opt, 'blocksize') && ~isempty(opt.blocksize)
        nblock = opt.blocksize;
    end

    % Call Fortran MEX
    [x, flag, relres, iter] = blqmr_(A, B, qtol, maxit, ...
                                      pcond_type, droptol_f, nblock);
    resv = [];  % Fortran backend does not return per-iteration residuals
    return
end

%% check input options

isprecond = 0;
if (nargin < 3 || (nargin >= 3 && isempty(qtol)))
    qtol = 1e-6;
end

if (nargin < 4 || (nargin >= 4 && isempty(maxit)))
    maxit = min(n, 20);
end

% parse opt for residual option
isquasires = 1;
if ~isempty(opt) && isfield(opt, 'residual')
    isquasires = ~opt.residual;
end

% auto-create preconditioner if opt.precond is set and M1/M2 not provided
if ~isempty(opt) && isfield(opt, 'precond') && ...
   (nargin < 5 || isempty(M1)) && (nargin < 6 || isempty(M2))
    [M1, M2] = create_preconditioner(A, opt);
end

% initialize preconditioner
if (nargin >= 5 && ~isempty(M1))
    if (nargin >= 6 && ~isempty(M2))
        isprecond = 2;
        % Split preconditioning: will solve M1^{-1}*A*M2^{-1}*y = M1^{-1}*b
        % and recover x = M2^{-1}*y at the end
    else
        M = M1;
        isprecond = 1;
    end
    % Check if M1 is a function handle (for custom preconditioners)
    if isa(M1, 'function_handle')
        isprecond = 3;  % Function handle preconditioner
        precond_func = M1;
    end
else
    if (nargin >= 6 && ~isempty(M2))
        fprintf(1, 'M2 is ignored as M1 is not given');
    end
end

% initialize initial guess
if (~exist('x0', 'var') || isempty(x0))
    x0 = zeros(size(B));
end

%% Optimized scalar path for m=1 (single RHS)
%  This provides ~2x speedup over MATLAB's built-in qmr()
if m == 1
    if (nargin < 6)
        M1 = [];
        M2 = [];
    end
    [x, flag, relres, iter, resv] = blqmr_scalar(A, B, qtol, maxit, ...
                                                 isprecond, M1, M2, x0, isquasires);
    return
end

%% Original block QMR algorithm for m>1

x = x0;
iter = 0;
relres = 1;

znm1 = zeros(n, m);
znm3 = zeros(n, m, 3);
zmm1 = zeros(m, m);
zmm3 = zeros(m, m, 3);

[v,   vt,  beta, alpha, omega, theta, Qa,  Qb,  Qc,  Qd] = deal( ...
                                                                znm3, znm1, zmm3, zmm1, zmm3, zmm1, zmm3, zmm3, zmm3, zmm3);

[zeta, zetat, eta, tau, taot, p] = deal(zmm1, zmm1, zmm1, zmm1, zmm1, znm3);

[t3, t3n, t3p] = updateidx(0);

Qa(:, :, t3) = eye(m);   % eq (35) in [Boyse1996], same below
Qd(:, :, t3n) = eye(m);
Qd(:, :, t3) = eye(m);

% vt0 contains the initial true residual
if (isprecond)
    if (isprecond == 3)
        vt = precond_func(B - A * x0);
    elseif (isprecond == 2)
        % Split preconditioning: transform to preconditioned space
        % We solve M1^{-1}*A*M2^{-1}*y = M1^{-1}*b where y = M2*x
        % Initial y0 = M2*x0, initial residual = M1^{-1}*b - M1^{-1}*A*M2^{-1}*y0
        %                                      = M1^{-1}*(b - A*x0)
        vt = M1 \ (B - A * x0);
    else
        vt = M \ (B - A * x0);
    end
    if (any(isnan(vt(:))))
        flag = 2;
        resv = [];
        if (nargout <= 1)
            fprintf('the preconditioner is rank-deficient, blqmr stops\n');
        end
        return
    end
else
    vt = B - A * x0;
end

[Q, R] = qqr(vt, 0); % quasi-qr decomposition,    %eq (37)
v(:, :, t3p) = Q;
beta(:, :, t3p) = R;

omega(:, :, t3p) = diag(sqrt(sum(conj(Q) .* Q, 1))); % eq (38)
taot = omega(:, :, t3p) * beta(:, :, t3p);            % eq (39)

% calculate the initial residual
if (isquasires)
    Qres0 = max(sqrt(sum(conj(taot) .* taot, 1)));  % eq (31)
else
    omegat = Q * diag(1 ./ diag(omega(:, :, t3p)));
    % R=(omegat*taot);
    % proof R==vt: omegat*taot=Q*inv(omega)*omega*beta=Q*R=vt
    Qres0 = max(sqrt(sum(conj(vt) .* vt, 1)));      % eq (32)
end

%% start iterative solving process

flag = 1;
resv = zeros(maxit, 1);

for k = 1:maxit
    [t3, t3n, t3p, t3nn] = updateidx(k);
    if (isprecond)
        if (isprecond == 3)
            vt = precond_func(A * v(:, :, t3)) - v(:, :, t3n) * beta(:, :, t3).';
        elseif (isprecond == 2)
            % Split preconditioning: apply M1^{-1} * A * M2^{-1}
            tmp = M2 \ v(:, :, t3);      % M2^{-1} * v
            tmp = A * tmp;              % A * M2^{-1} * v
            tmp = M1 \ tmp;             % M1^{-1} * A * M2^{-1} * v
            vt = tmp - v(:, :, t3n) * beta(:, :, t3).';
        else
            vt = M \ (A * v(:, :, t3)) - v(:, :, t3n) * beta(:, :, t3).';
        end
    else
        vt = (A * v(:, :, t3)) - v(:, :, t3n) * beta(:, :, t3).';
    end
    alpha = v(:, :, t3).' * vt;  % eq (41)
    vt = vt - v(:, :, t3) * alpha; % eq (42)

    % Check for quasi-norm breakdown (can happen in complex symmetric systems)
    % quasi-norm = sqrt(sum(vt.*vt)) can be zero even if ||vt||_2 is not
    vt_quasi_norm_sq = sum(vt .* vt, 1);  % No conjugate - quasi inner product
    if any(abs(vt_quasi_norm_sq) < 1e-28)
        % Near breakdown - algorithm has essentially converged or stagnated
        flag = 3;
        iter = k;
        break
    end

    [Q, R] = qqr(vt, 0);       % eq (43), quasi-qr decomposition
    v(:, :, t3p) = Q;
    beta(:, :, t3p) = R;
    omega(:, :, t3p) = diag(sqrt(sum(conj(Q) .* Q, 1))); % eq (44)
    tmp0 = omega(:, :, t3n) * beta(:, :, t3).';
    theta = Qb(:, :, t3nn) * tmp0; % eq (45)
    tmp1 = Qd(:, :, t3nn) * tmp0;
    tmp2 = omega(:, :, t3) * alpha;
    eta =  Qa(:, :, t3n) * tmp1 + Qb(:, :, t3n) * tmp2; % eq (46)
    zetat = Qc(:, :, t3n) * tmp1 + Qd(:, :, t3n) * tmp2; % eq (47)

    [QQ, zeta] = qr([zetat; omega(:, :, t3p) * beta(:, :, t3p)]);  % eq (48)
    zeta = zeta(1:m, 1:m);
    QQ = QQ'; % QQ is an orthogonal matrix, thus, Q'=inv(Q), here is 1 of the 2
    % places need congugate transpose

    Qa(:, :, t3) = QQ(1:m, 1:m);
    Qb(:, :, t3) = QQ(1:m, m + 1:2 * m);
    Qc(:, :, t3) = QQ(m + 1:2 * m, 1:m);
    Qd(:, :, t3) = QQ(m + 1:2 * m, m + 1:2 * m);

    % Check zeta before inversion to detect breakdown
    zeta_diag = diag(zeta);
    if any(abs(zeta_diag) < 1e-14) || any(~isfinite(zeta_diag))
        flag = 3;  % stagnated/breakdown
        iter = k;
        break
    end

    % Use pseudo-inverse if zeta is ill-conditioned (like Python version)
    zeta_rcond = rcond(zeta);
    if zeta_rcond < 1e-14
        % Fall back to pseudo-inverse for robustness
        p(:, :, t3) = (v(:, :, t3) - p(:, :, t3n) * eta - p(:, :, t3nn) * theta) * pinv(zeta);
    else
        p(:, :, t3) = (v(:, :, t3) - p(:, :, t3n) * eta - p(:, :, t3nn) * theta) / zeta; % eq (49)
    end

    tau = Qa(:, :, t3) * taot;  % eq (50)
    x = x + p(:, :, t3) * tau;    % eq (51)
    taot = Qc(:, :, t3) * taot; % eq (52)

    % calculate residual and terminate if converge
    if (isquasires)
        Qres = max(sqrt(sum(conj(taot) .* taot, 1))); % eq (31)
    else
        omegat = omegat * Qc(:, :, t3)' + v(:, :, t3p) * (Qd(:, :, t3) * diag(1 ./ diag(omega(:, :, t3p))))';
        R = omegat * taot; % R is the residual, R==B-A*x;
        Qres = max(sqrt(sum(conj(R) .* R, 1))); % error is the max column norm
    end
    resv(k) = Qres;
    if (k > 1 && Qres == Qres1)
        flag = 3; % stagnated
        iter = k;
        break
    end
    Qres1 = Qres;
    relres = Qres / Qres0;
    iter = k;
    if (relres <= qtol)
        flag = 0;
        if (nargout <= 1)
            fprintf(['blqmr converged at iteration %d with a relative ' ...
                     'quasi-residual %e\n'], iter, relres);
        end
        break
    end
end
resv = resv(1:k);

if ((flag == 1 || flag == 3) && nargout <= 1)
    resname = 'quasi-';
    if (isquasires == 0)
        resname = '';
    end
    warning(['blqmr failed to converge within %d iterations,\nthe final'...
             ' %sresidual was %e\n'], iter, resname, relres);
end

% For split preconditioning, recover x = M2^{-1} * y
% The iteration computed y where y = M2*x, so x = M2\y
if (isprecond == 2)
    x = M2 \ x;
end

%% ========================================================================
%  Helper functions
%% ========================================================================

%% create preconditioner from opt.precond
function [M1, M2] = create_preconditioner(A, opt)
M1 = [];
M2 = [];

precond_type = opt.precond;
if isstruct(precond_type)
    ptype = precond_type.type;
    popts = precond_type;
else
    ptype = precond_type;
    popts = opt;
end

droptol = 1e-3;
if isfield(popts, 'droptol')
    droptol = popts.droptol;
end

is_complex = ~isreal(A);

try
    switch lower(ptype)
        case 'ilu'
            % ILU(0) - real matrices only, falls back to diagonal for complex
            if is_complex
                [M1, M2] = diag_precond(A);
            else
                [M1, M2] = ilu(A, struct('type', 'nofill'));
            end

        case {'ilutp', 'ilut'}
            % ILUTP with threshold - real matrices only
            if is_complex
                [M1, M2] = diag_precond(A);
            else
                [M1, M2] = ilu(A, struct('type', 'ilutp', 'droptol', droptol));
            end

        case 'ichol'
            % Incomplete Cholesky - for SPD matrices only
            if is_complex
                [M1, M2] = diag_precond(A);
            else
                ichol_opts = struct('type', 'ict', 'droptol', droptol);
                if isfield(popts, 'michol')
                    ichol_opts.michol = popts.michol;
                end
                L = ichol(A, ichol_opts);
                M1 = L;
                M2 = L';
            end

        case 'diag'
            % Diagonal (Jacobi) preconditioner - works for all matrices
            [M1, M2] = diag_precond(A);

        otherwise
            warning('Unknown preconditioner type: %s, using none', ptype);
    end
catch ME
    warning('Preconditioner creation failed: %s\nFalling back to diagonal', ME.message);
    try
        [M1, M2] = diag_precond(A);
    catch
        warning('Diagonal preconditioner also failed, using none');
    end
end

%% diagonal (Jacobi) preconditioner
function [M1, M2] = diag_precond(A)
d = diag(A);
% Handle zero or near-zero diagonal entries
small_thresh = max(abs(d)) * 1e-14;
if small_thresh == 0
    small_thresh = 1e-14;
end
small_idx = abs(d) < small_thresh;
d(small_idx) = 1;
M1 = spdiags(1 ./ d, 0, size(A, 1), size(A, 1));
M2 = [];

%% cyclic array index
function tnew = cycidx(t, dim)
tnew = mod(t, dim) + 1;

%% compute the cyclic indices for all internal arrays
function [t3, t3n, t3p, t3nn] = updateidx(t)
t3 = cycidx(t, 3);    % step k
t3n = cycidx(t - 1, 3); % step k-1
t3p = cycidx(t + 1, 3); % step k+1
t3nn = cycidx(t - 2, 3); % step k-2

%% ========================================================================
%  Optimized scalar QMR for single RHS (m=1)
%  Uses same algorithm as block version but with scalar operations
%% ========================================================================
function [x, flag, relres, iter, resv] = blqmr_scalar(A, B, qtol, maxit, ...
                                                      isprecond, M1, M2, x0, isquasires)

n = length(B);
x = x0;

% Cyclic index helper (same as block version)
cyc = @(t) mod(t, 3) + 1;

% Initialize arrays with cyclic storage (3 slots each)
v = zeros(n, 3);      % Lanczos vectors
beta = zeros(1, 3);   % quasi-norms
omega = zeros(1, 3);  % 2-norms
p = zeros(n, 3);      % search directions

% Q matrix components (2x2 becomes 4 scalars per time step)
Qa = zeros(1, 3);
Qb = zeros(1, 3);
Qc = zeros(1, 3);
Qd = zeros(1, 3);

% Initial indices
t3 = cyc(0);
t3n = cyc(-1);
t3p = cyc(1);

% Initialize Q matrices (eq 35 in Boyse1996)
Qa(t3) = 1;
Qd(t3n) = 1;
Qd(t3) = 1;

% Compute initial residual with preconditioning
r0 = B - A * x0;
if isprecond == 2
    vt = M1 \ r0;
elseif isprecond == 1
    vt = M1 \ r0;
elseif isprecond == 3
    vt = M1(r0);
else
    vt = r0;
end

if any(~isfinite(vt))
    flag = 2;
    relres = 1;
    iter = 0;
    resv = [];
    return
end

% Quasi-QR of initial residual (eq 37)
beta(t3p) = sqrt(vt.' * vt);  % quasi-norm
if abs(beta(t3p)) < 1e-30
    flag = 0;
    relres = 0;
    iter = 0;
    resv = [];
    return
end
v(:, t3p) = vt / beta(t3p);

% omega = 2-norm of quasi-orthonormal vector (eq 38)
omega(t3p) = sqrt(v(:, t3p)' * v(:, t3p));

% taot (eq 39)
taot = omega(t3p) * beta(t3p);

% Initial residual norm
if isquasires
    Qres0 = abs(taot);
else
    omegat = v(:, t3p) / omega(t3p);
    Qres0 = norm(vt);
end

if Qres0 < 1e-30
    flag = 0;
    relres = 0;
    iter = 0;
    resv = [];
    return
end

flag = 1;
resv = zeros(maxit, 1);
Qres_prev = inf;

for k = 1:maxit
    % Update cyclic indices
    t3 = cyc(k);
    t3n = cyc(k - 1);
    t3p = cyc(k + 1);
    t3nn = cyc(k - 2);

    % Lanczos step (eq 40-42)
    if isprecond == 2
        tmp = M2 \ v(:, t3);
        tmp = A * tmp;
        vt = M1 \ tmp - v(:, t3n) * beta(t3);
    elseif isprecond == 1
        vt = M1 \ (A * v(:, t3)) - v(:, t3n) * beta(t3);
    elseif isprecond == 3
        vt = M1(A * v(:, t3)) - v(:, t3n) * beta(t3);
    else
        vt = A * v(:, t3) - v(:, t3n) * beta(t3);
    end

    alpha = v(:, t3).' * vt;  % eq 41
    vt = vt - v(:, t3) * alpha;  % eq 42

    % Check for breakdown
    vt_qnorm_sq = vt.' * vt;
    if abs(vt_qnorm_sq) < 1e-28
        flag = 3;
        iter = k;
        break
    end

    % Quasi-QR (eq 43)
    beta(t3p) = sqrt(vt_qnorm_sq);
    v(:, t3p) = vt / beta(t3p);
    omega(t3p) = sqrt(v(:, t3p)' * v(:, t3p));  % eq 44

    % Build quantities for QR (eqs 45-47)
    tmp0 = omega(t3n) * beta(t3);
    theta = Qb(t3nn) * tmp0;  % eq 45
    tmp1 = Qd(t3nn) * tmp0;
    tmp2 = omega(t3) * alpha;
    eta = Qa(t3n) * tmp1 + Qb(t3n) * tmp2;  % eq 46
    zetat = Qc(t3n) * tmp1 + Qd(t3n) * tmp2;  % eq 47

    % QR factorization of [zetat; omega(t3p)*beta(t3p)] (eq 48)
    % For m=1, this is a Givens rotation
    rhs = omega(t3p) * beta(t3p);
    zeta = sqrt(abs(zetat)^2 + abs(rhs)^2);

    if zeta < 1e-14
        flag = 3;
        iter = k;
        break
    end

    % Givens rotation: Q' * [zetat; rhs] = [zeta; 0]
    % Q = [c s; -s' c'] where c = zetat/zeta, s = rhs/zeta
    c = zetat / zeta;
    s = rhs / zeta;

    % Q' = [c' -s; s' c] maps to Qa,Qb,Qc,Qd
    Qa(t3) = conj(c);
    Qb(t3) = conj(s);
    Qc(t3) = -s;
    Qd(t3) = c;

    % Update p (eq 49)
    p(:, t3) = (v(:, t3) - p(:, t3n) * eta - p(:, t3nn) * theta) / zeta;

    % Update solution (eqs 50-52)
    tau = Qa(t3) * taot;  % eq 50
    x = x + p(:, t3) * tau;  % eq 51
    taot = Qc(t3) * taot;  % eq 52

    % Compute residual
    if isquasires
        Qres = abs(taot);  % eq 31
    else
        omegat = omegat * Qc(t3)' + v(:, t3p) * (Qd(t3) / omega(t3p));
        R = omegat * taot;
        Qres = abs(R);  % eq 32
    end

    resv(k) = Qres;

    % Check for stagnation
    if k > 1 && Qres == Qres_prev
        flag = 3;
        iter = k;
        break
    end
    Qres_prev = Qres;

    relres = Qres / Qres0;
    iter = k;

    if relres <= qtol
        flag = 0;
        break
    end
end

resv = resv(1:iter);

% Recover solution for split preconditioning
if isprecond == 2
    x = M2 \ x;
end