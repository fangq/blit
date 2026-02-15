function blqmr_compare()
% BLQMR_COMPARE Compare Fortran MEX vs native MATLAB blqmr solver
%
%   Tests convergence, accuracy, and speed for real and complex
%   symmetric sparse systems of increasing size.

fprintf('=====================================================================\n');
fprintf('BLQMR: Fortran MEX vs Native MATLAB Comparison\n');
fprintf('=====================================================================\n\n');

has_mex = exist('blqmr_', 'file') == 3;
if ~has_mex
    error('Fortran MEX blqmr_ not found. Build it first: cd matlab/src && make');
end

% Print thread info
fprintf('Environment:\n');
% fprintf('  MATLAB threads (maxNumCompThreads): %d\n', maxNumCompThreads);
try
    nthreads_omp = str2double(getenv('OMP_NUM_THREADS'));
    if isnan(nthreads_omp)
        fprintf('  OMP_NUM_THREADS: (not set)\n');
    else
        fprintf('  OMP_NUM_THREADS: %d\n', nthreads_omp);
    end
catch
    fprintf('  OMP_NUM_THREADS: (not set)\n');
end
fprintf('  MKL_NUM_THREADS: %s\n', getenv_or(getenv('MKL_NUM_THREADS'), '(not set)'));
fprintf('  OPENBLAS_NUM_THREADS: %s\n', getenv_or(getenv('OPENBLAS_NUM_THREADS'), '(not set)'));
try
    [~, ncpu] = system('nproc');
    fprintf('  CPU cores: %s', ncpu);
catch
end
fprintf('\n');

% Test parameters
tol = 1e-8;
maxiter = 2000;
nruns = 3;

% =====================================================================
% Test 1: Real SPD, single RHS
% =====================================================================
fprintf('--- Test 1: Real SPD (n~4000, nrhs=1) ---\n');
if exist('rng', 'file')
    rng(42);
else
    rand('state', 42);
    randn('state', 42);
end
A = gallery('poisson', 64);  % 4096x4096
n = size(A, 1);
b = randn(n, 1);
run_comparison(A, b, tol, maxiter, nruns, 0);

% =====================================================================
% Test 2: Real SPD, 4 RHS
% =====================================================================
fprintf('\n--- Test 2: Real SPD (n~4000, nrhs=4) ---\n');
B = randn(n, 4);
run_comparison(A, B, tol, maxiter, nruns, 0);

% =====================================================================
% Test 3: Real SPD, 8 RHS
% =====================================================================
fprintf('\n--- Test 3: Real SPD (n~4000, nrhs=8) ---\n');
B = randn(n, 8);
run_comparison(A, B, tol, maxiter, nruns, 0);

% =====================================================================
% Test 4: Larger real SPD, single RHS
% =====================================================================
fprintf('\n--- Test 4: Larger real SPD (n~16000, nrhs=1) ---\n');
A2 = gallery('poisson', 128);  % 16384x16384
n2 = size(A2, 1);
b2 = randn(n2, 1);
run_comparison(A2, b2, tol, maxiter, nruns, 0);

% =====================================================================
% Test 5: Larger real SPD, 4 RHS
% =====================================================================
fprintf('\n--- Test 5: Larger real SPD (n~16000, nrhs=4) ---\n');
B2 = randn(n2, 4);
run_comparison(A2, B2, tol, maxiter, nruns, 0);

% =====================================================================
% Test 6: Larger real SPD, 8 RHS
% =====================================================================
fprintf('\n--- Test 6: Larger real SPD (n~16000, nrhs=8) ---\n');
B2 = randn(n2, 8);
run_comparison(A2, B2, tol, maxiter, nruns, 0);

% =====================================================================
% Test 7: Complex symmetric, single RHS
% =====================================================================
fprintf('\n--- Test 7: Complex symmetric (n~4000, nrhs=1) ---\n');
Ac = A + 0.05i * sprandsym(n, nnz(A) / n^2);
Ac = (Ac + Ac.') / 2 + 20 * speye(n);  % complex symmetric
bc = randn(n, 1) + 1i * randn(n, 1);
run_comparison(Ac, bc, tol, maxiter, nruns, 0);

% =====================================================================
% Test 8: Complex symmetric, 4 RHS
% =====================================================================
fprintf('\n--- Test 8: Complex symmetric (n~4000, nrhs=4) ---\n');
Bc = randn(n, 4) + 1i * randn(n, 4);
run_comparison(Ac, Bc, tol, maxiter, nruns, 0);

% =====================================================================
% Test 9: Large complex symmetric, single RHS
% =====================================================================
fprintf('\n--- Test 9: Large complex symmetric (n~16000, nrhs=1) ---\n');
Ac2 = A2 + 0.05i * sprandsym(n2, nnz(A2) / n2^2);
Ac2 = (Ac2 + Ac2.') / 2 + 20 * speye(n2);
bc2 = randn(n2, 1) + 1i * randn(n2, 1);
run_comparison(Ac2, bc2, tol, maxiter, nruns, 0);

% =====================================================================
% Test 10: Large complex symmetric, 4 RHS
% =====================================================================
fprintf('\n--- Test 10: Large complex symmetric (n~16000, nrhs=4) ---\n');
Bc2 = randn(n2, 4) + 1i * randn(n2, 4);
run_comparison(Ac2, Bc2, tol, maxiter, nruns, 0);

% =====================================================================
% Test 11: Jacobi precond, real SPD
% =====================================================================
fprintf('\n--- Test 11: Real SPD + Jacobi (n~4000, nrhs=4) ---\n');
B = randn(n, 4);
run_comparison(A, B, tol, maxiter, nruns, 3);

% =====================================================================
% Test 12: Jacobi precond, complex symmetric
% =====================================================================
fprintf('\n--- Test 12: Complex symmetric + Jacobi (n~4000, nrhs=4) ---\n');
run_comparison(Ac, Bc, tol, maxiter, nruns, 3);

% =====================================================================
% Test 13: Very large real SPD
% =====================================================================
fprintf('\n--- Test 13: Very large real SPD (n~65000, nrhs=1) ---\n');
A3 = gallery('poisson', 256);  % 65536x65536
n3 = size(A3, 1);
b3 = randn(n3, 1);
run_comparison(A3, b3, tol, maxiter, nruns, 0);

% =====================================================================
% Test 14: Very large real SPD, multiple RHS
% =====================================================================
fprintf('\n--- Test 14: Very large real SPD (n~65000, nrhs=4) ---\n');
B3 = randn(n3, 4);
run_comparison(A3, B3, tol, maxiter, nruns, 0);

fprintf('\n=====================================================================\n');
fprintf('All comparisons complete.\n');
fprintf('=====================================================================\n');
end

function run_comparison(A, B, tol, maxiter, nruns, pcond_type_mex)

n = size(A, 1);
nrhs = size(B, 2);
is_complex = ~isreal(A) || ~isreal(B);

precond_names = {'none', 'ILU-left', 'ILU-split', 'Jacobi'};
precond_name = precond_names{pcond_type_mex + 1};
fprintf('  n=%d, nrhs=%d, nnz=%d, complex=%d, precond=%s\n', ...
        n, nrhs, nnz(A), is_complex, precond_name);

% ----- Native MATLAB blqmr -----
opt_native = struct('usefortran', 0);
if pcond_type_mex == 3
    opt_native.precond = 'diag';
elseif pcond_type_mex > 0
    if is_complex
        opt_native.precond = 'diag';
    else
        opt_native.precond = 'ilu';
    end
end

times_native = zeros(nruns, 1);
for r = 1:nruns
    tic;
    [x_nat, flag_nat, relres_nat, iter_nat] = blqmr(A, B, tol, maxiter, [], [], [], opt_native);
    times_native(r) = toc;
end
t_native = median(times_native);

% ----- Fortran MEX blqmr_ -----
pcond_type = pcond_type_mex;

times_mex = zeros(nruns, 1);
x_mex = [];
flag_mex = -1;
relres_mex = Inf;
iter_mex = 0;
t_mex = 0;
mex_ok = true;
try
    for r = 1:nruns
        tic;
        [x_mex, flag_mex, relres_mex, iter_mex] = blqmr_(A, B, tol, maxiter, ...
                                                         pcond_type, 0.001, 0);
        times_mex(r) = toc;
    end
    t_mex = median(times_mex);
catch me
    mex_ok = false;
    fprintf('  MEX FAILED: %s\n', me.message);
end

if ~mex_ok
    return
end

% ----- Compute true residuals -----
if flag_nat <= 1 && ~any(isnan(x_nat(:)))
    true_res_nat = max_col_relres(A, x_nat, B);
else
    true_res_nat = Inf;
end

if flag_mex <= 1 && ~any(isnan(x_mex(:)))
    true_res_mex = max_col_relres(A, x_mex, B);
else
    true_res_mex = Inf;
end

% ----- Solution agreement -----
if flag_nat <= 1 && flag_mex <= 1
    sol_diff = norm(x_nat(:) - x_mex(:)) / max(norm(x_nat(:)), 1e-16);
else
    sol_diff = Inf;
end

% ----- Report -----
fprintf('  %-12s %8s %6s %6s %12s %12s %10s\n', '', 'Time', 'Flag', 'Iter', 'RelRes', 'TrueRes', 'Speedup');
fprintf('  %-12s %8s %6s %6s %12s %12s %10s\n', '', '--------', '------', '------', '------------', '------------', '----------');

fprintf('  %-12s %7.1fms %6d %6d %12.2e %12.2e %10s\n', ...
        'Native', t_native * 1000, flag_nat, iter_nat, relres_nat, true_res_nat, '-');

if t_mex > 0 && t_native > 0
    speedup = t_native / t_mex;
    speedup_str = sprintf('%.2fx', speedup);
else
    speedup_str = 'N/A';
end

fprintf('  %-12s %7.1fms %6d %6d %12.2e %12.2e %10s\n', ...
        'Fortran MEX', t_mex * 1000, flag_mex, iter_mex, relres_mex, true_res_mex, speedup_str);

if sol_diff < Inf
    fprintf('  Solution difference (relative): %.2e\n', sol_diff);
end

% Check agreement
if flag_nat == 0 && flag_mex == 0
    if abs(iter_nat - iter_mex) <= 2 && sol_diff < 1e-4
        fprintf('  Result: AGREE (same convergence)\n');
    elseif sol_diff < 1e-2
        fprintf('  Result: CLOSE (minor differences, iter diff=%d)\n', abs(iter_nat - iter_mex));
    else
        fprintf('  Result: DIFFER (flag_nat=%d, flag_mex=%d, sol_diff=%.2e)\n', ...
                flag_nat, flag_mex, sol_diff);
    end
elseif flag_nat ~= flag_mex
    fprintf('  Result: DIVERGENT FLAGS (native=%d, mex=%d)\n', flag_nat, flag_mex);
end

end

function res = max_col_relres(A, X, B)
% MAX_COL_RELRES Max relative residual across columns
res = 0;
for j = 1:size(B, 2)
    bnorm = norm(B(:, j));
    if bnorm > 0
        res = max(res, norm(A * X(:, j) - B(:, j)) / bnorm);
    end
end
end

function s = getenv_or(val, default)
% GETENV_OR Return environment variable or default string
if isempty(val)
    s = default;
else
    s = val;
end
end

function A = make_poisson_2d(m)
% MAKE_POISSON_2D  m^2 x m^2 2D Poisson matrix (5-point stencil)
%   Octave-compatible replacement for gallery('poisson', m)
e = ones(m, 1);
T = spdiags([-e 2 * e -e], [-1 0 1], m, m);
I = speye(m);
A = kron(I, T) + kron(T, I);
end
