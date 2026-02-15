function blqmr_bench_parallel()
% BLQMR_BENCH_PARALLEL Compare native MATLAB, single-thread Fortran, and
%   OpenMP Fortran BLQMR on complex symmetric Helmholtz FEM problems.
%
%   Tests: 1) Native MATLAB blqmr (opt.usefortran=0)
%          2) Fortran MEX, single block (nblock=0)
%          3) Fortran MEX, OpenMP parallel (nblock=1, 2, 4)

fprintf('=====================================================================\n');
fprintf('BLQMR Parallel Benchmark: Native vs Fortran vs OpenMP\n');
fprintf('=====================================================================\n\n');

has_mex = exist('blqmr_', 'file') == 3;
if ~has_mex
    % Octave: check for .mex as well
    has_mex = exist('blqmr_', 'file') == 2 && ~isempty(strfind(which('blqmr_'), '.mex'));
end
if ~has_mex
    error('Fortran MEX blqmr_ not found. Build it first: cd matlab/src && make');
end

is_octave = exist('OCTAVE_VERSION', 'builtin') > 0;

% Print environment
fprintf('Environment:\n');
if is_octave
    fprintf('  Platform: GNU Octave %s\n', OCTAVE_VERSION);
    fprintf('  Comp threads: N/A (Octave)\n');
else
    fprintf('  Platform: MATLAB %s\n', version);
    fprintf('  Comp threads: %d\n', maxNumCompThreads);
end
try
    [~, c] = system('nproc');
    fprintf('  CPU cores: %s', c);
catch
end
fprintf('\n');

% Configuration
grid_sizes = [10, 15, 20, 25];
rhs_counts = [4, 16, 64, 128];
nblock_vals = [0, 1, 2, 4];  % 0=single block, 1/2/4=OMP chunk sizes
tol = 1e-6;
maxiter = 2000;
n_runs = 2;

fprintf('Configuration:\n');
fprintf('  Grid sizes: %s (nodes = grid^3)\n', mat2str(grid_sizes));
fprintf('  RHS counts: %s\n', mat2str(rhs_counts));
fprintf('  OMP nblock: %s (0=all-at-once, N=N RHS per thread)\n', mat2str(nblock_vals));
fprintf('  Tolerance: %.0e, maxiter: %d, runs: %d (min time)\n\n', tol, maxiter, n_runs);

% Preconditioner type for Fortran: 3 = Jacobi-split (safe, no UMFPACK/MKL conflict)
pcond_type = 3;
droptol = 0.001;

for gi = 1:length(grid_sizes)
    gridsize = grid_sizes(gi);

    % Build mesh and matrix
    [node, elem] = meshgrid6(0:gridsize - 1, 0:gridsize - 1, 0:gridsize - 1);
    n = size(node, 1);
    A = assemble_helmholtz_fem(node, elem);

    fprintf('=====================================================================\n');
    fprintf('Grid %d^3: n=%d, nnz=%d, complex symmetric Helmholtz\n', gridsize, n, nnz(A));
    fprintf('=====================================================================\n');

    % Jacobi preconditioner for native MATLAB path
    [M1_jac, M2_jac] = create_split_jacobi(A);

    for ri = 1:length(rhs_counts)
        nrhs = rhs_counts(ri);

        % Skip combinations that would be too slow
        if n * nrhs > 5e7
            fprintf('\n  RHS=%d: SKIPPED (too large)\n', nrhs);
            continue
        end

        fprintf('\n  RHS=%d\n', nrhs);
        fprintf('  %-28s %10s %6s %6s %12s %10s\n', ...
                'Method', 'Time', 'Flag', 'Iter', 'TrueRes', 'Speedup');
        fprintf('  %-28s %10s %6s %6s %12s %10s\n', ...
                repmat('-', 1, 28), '----------', '------', '------', '------------', '----------');

        % Generate RHS
        if exist('rng', 'file')
            rng(42);
        else
            rand('state', 42);
            randn('state', 42);
        end
        B = create_distributed_sources(node, elem, nrhs);

        % Reference: direct solver
        t_direct = bench_direct(A, B, n_runs);

        % ===== 1) Native MATLAB blqmr =====
        opt_nat = struct('usefortran', 0, 'precond', 'diag');
        times_nat = zeros(n_runs, 1);
        for r = 1:n_runs
            tic;
            [x_nat, flag_nat, ~, iter_nat] = blqmr(A, B, tol, maxiter, [], [], [], opt_nat);
            times_nat(r) = toc;
        end
        t_nat = min(times_nat);
        res_nat = safe_residual(A, x_nat, B, flag_nat);
        print_row('Native MATLAB', t_nat, flag_nat, iter_nat, res_nat, t_direct);

        % ===== 2) Fortran MEX, single block (nblock=0) =====
        times_f1 = zeros(n_runs, 1);
        for r = 1:n_runs
            tic;
            [x_f1, flag_f1, ~, iter_f1] = blqmr_(A, B, tol, maxiter, pcond_type, droptol, 0);
            times_f1(r) = toc;
        end
        t_f1 = min(times_f1);
        res_f1 = safe_residual(A, x_f1, B, flag_f1);
        print_row('Fortran (nblock=0)', t_f1, flag_f1, iter_f1, res_f1, t_direct);

        % ===== 3) Fortran MEX, OpenMP with various nblock =====
        for ni = 1:length(nblock_vals)
            nb = nblock_vals(ni);
            if nb == 0
                continue
            end  % already tested above
            if nb >= nrhs
                continue
            end  % no splitting needed

            times_omp = zeros(n_runs, 1);
            for r = 1:n_runs
                tic;
                [x_omp, flag_omp, ~, iter_omp] = blqmr_(A, B, tol, maxiter, pcond_type, droptol, nb);
                times_omp(r) = toc;
            end
            t_omp = min(times_omp);
            res_omp = safe_residual(A, x_omp, B, flag_omp);
            label = sprintf('Fortran OMP (nblock=%d)', nb);
            print_row(label, t_omp, flag_omp, iter_omp, res_omp, t_direct);
        end

        % Reference line
        fprintf('  %-28s %9.1fms %6s %6s %12s %10s\n', ...
                'Direct (mldivide)', t_direct * 1000, '-', '-', '-', '1.00x');
    end
    fprintf('\n');
end

fprintf('=====================================================================\n');
fprintf('Done.\n');
fprintf('=====================================================================\n');
end

%% ========================================================================

function print_row(label, t, flag, iter, res, t_ref)
if t > 0 && t_ref > 0
    sp = sprintf('%.2fx', t_ref / t);
else
    sp = 'N/A';
end
if res < Inf
    res_str = sprintf('%.2e', res);
else
    res_str = 'FAILED';
end
fprintf('  %-28s %9.1fms %6d %6d %12s %10s\n', ...
        label, t * 1000, flag, iter, res_str, sp);
end

function t = bench_direct(A, B, n_runs)
times = zeros(n_runs, 1);
for r = 1:n_runs
    tic;
    A \ B;
    times(r) = toc;
end
t = min(times);
end

function res = safe_residual(A, X, B, flag)
if flag > 1 || isempty(X) || any(isnan(X(:)))
    res = Inf;
    return
end
res = 0;
for j = 1:size(B, 2)
    bn = norm(B(:, j));
    if bn > 0
        res = max(res, norm(A * X(:, j) - B(:, j)) / bn);
    end
end
end

%% ========================================================================
%  FEM mesh and matrix assembly
%% ========================================================================

function [node, elem] = meshgrid6(x, y, z)
nx = length(x);
ny = length(y);
nz = length(z);
[X, Y, Z] = ndgrid(x, y, z);
node = [X(:), Y(:), Z(:)];
elem = [];
idx = @(i, j, k) (i - 1) * ny * nz + (j - 1) * nz + k;
for i = 1:nx - 1
    for j = 1:ny - 1
        for k = 1:nz - 1
            n0 = idx(i, j, k);
            n1 = idx(i + 1, j, k);
            n2 = idx(i + 1, j + 1, k);
            n3 = idx(i, j + 1, k);
            n4 = idx(i, j, k + 1);
            n5 = idx(i + 1, j, k + 1);
            n6 = idx(i + 1, j + 1, k + 1);
            n7 = idx(i, j + 1, k + 1);
            elem = [elem; n0 n1 n3 n4; n1 n2 n3 n6; n1 n4 n5 n6
                    n3 n4 n6 n7; n1 n3 n4 n6; n1 n2 n6 n5];
        end
    end
end
end

function A = assemble_helmholtz_fem(node, elem)
n = size(node, 1);
nelem = size(elem, 1);
max_entries = 16 * nelem;
II = zeros(max_entries, 1);
JJ = zeros(max_entries, 1);
VV = zeros(max_entries, 1);
cnt = 0;
for e = 1:nelem
    id = elem(e, :);
    co = node(id, :);
    d1 = co(2, :) - co(1, :);
    d2 = co(3, :) - co(1, :);
    d3 = co(4, :) - co(1, :);
    J = [d1; d2; d3]';
    vol = abs(det(J)) / 6;
    if vol < 1e-15
        continue
    end
    iJ = inv(J);
    gr = [-1 -1 -1; 1 0 0; 0 1 0; 0 0 1]';
    gN = iJ' * gr;
    Ke = vol * (gN' * gN);
    Me = vol / 20 * (ones(4) + eye(4));
    Ae = Ke - 1.0 * Me + 0.3i * Me;
    for i = 1:4
        for j = 1:4
            cnt = cnt + 1;
            II(cnt) = id(i);
            JJ(cnt) = id(j);
            VV(cnt) = Ae(i, j);
        end
    end
end
A = sparse(II(1:cnt), JJ(1:cnt), VV(1:cnt), n, n);
A = (A + A.') / 2 + 0.01 * mean(abs(diag(A))) * speye(n);
end

function B = create_distributed_sources(node, elem, nrhs)
n = size(node, 1);
B = zeros(n, nrhs);
xr = [min(node(:, 1)), max(node(:, 1))];
yr = [min(node(:, 2)), max(node(:, 2))];
zr = [min(node(:, 3)), max(node(:, 3))];
ns = max(1, ceil(nrhs^(1 / 3)));
src_pos = [];
for iz = 1:ns
    for iy = 1:ns
        for ix = 1:ns
            if size(src_pos, 1) >= nrhs
                break
            end
            fx = 0.15 + 0.7 * (ix - 0.5) / ns;
            fy = 0.15 + 0.7 * (iy - 0.5) / ns;
            fz = 0.15 + 0.7 * (iz - 0.5) / ns;
            src_pos = [src_pos; xr(1) + fx * diff(xr), yr(1) + fy * diff(yr), zr(1) + fz * diff(zr)];
        end
    end
end
src_pos = src_pos(1:nrhs, :);
for k = 1:nrhs
    phase = exp(1i * 2 * pi * (k - 1) / max(nrhs, 1));
    [~, nearest] = min(sum((node - src_pos(k, :)).^2, 2));
    B(nearest, k) = phase;
end
end

function [M1, M2] = create_split_jacobi(A)
d = diag(A);
d(abs(d) < max(max(abs(d)) * 1e-14, 1e-14)) = 1.0;
sd = sqrt(d);
M1 = spdiags(sd, 0, length(d), length(d));
M2 = M1;
end
