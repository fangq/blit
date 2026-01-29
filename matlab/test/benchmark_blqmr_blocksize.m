function benchmark_blqmr_blocksize()
% BENCHMARK_BLQMR_BLOCKSIZE Test BLQMR speedup from batching RHS
%
%   Compares: mldivide, QMR, BLQMR with various block sizes
%   on complex symmetric FEM matrices
%   Uses split preconditioning (M1 = M2 = sqrt(D)) for Jacobi

fprintf('=====================================================================================================\n');
fprintf('BLQMR BENCHMARK: mldivide vs QMR vs BLQMR (block sizes 1-64)\n');
fprintf('=====================================================================================================\n\n');

total_rhs = 64;
block_sizes = [1:10, 12:2:20, 22:4:32, 40:8:64];
tol = 1e-6;
maxiter = 2000;
grid_sizes = [10, 20, 30];

fprintf('Config: %d RHS, tol=%.0e, maxiter=%d\n', total_rhs, tol, maxiter);
fprintf('Block sizes: %s\n', mat2str(block_sizes));
fprintf('Preconditioner: split Jacobi (M1 = M2 = sqrt(D))\n');
fprintf('Note: Remainder RHS handled when %d not divisible by block size\n\n', total_rhs);

all_results = cell(length(grid_sizes), 1);

for gi = 1:length(grid_sizes)
    m = grid_sizes(gi);
    fprintf('=====================================================================================================\n');
    fprintf('GRID %d^3\n', m);
    fprintf('=====================================================================================================\n');

    [node, elem] = meshgrid6(0:m - 1, 0:m - 1, 0:m - 1);
    n = size(node, 1);
    A = assemble_helmholtz_fem(node, elem);
    B = create_distributed_sources(node, elem, total_rhs);

    fprintf('  Matrix: n=%d, nnz=%d, complex symmetric\n', n, nnz(A));

    % Create split preconditioner for Jacobi (M1 = M2 = sqrt(D))
    use_precond = true;

    if use_precond
        fprintf('  Creating split preconditioner (sqrt diagonal)...');
        d = full(diag(A));
        small_thresh = max(abs(d)) * 1e-14;
        if small_thresh == 0
            small_thresh = 1e-14;
        end
        small_idx = abs(d) < small_thresh;
        d(small_idx) = 1;
        % Split preconditioning: M1 = M2 = sqrt(D)
        % This gives equilibration: D^{-1/2} * A * D^{-1/2}
        sqrt_d = sqrt(d);
        M1 = spdiags(sqrt_d, 0, n, n);
        M2 = M1;  % Same as M1 for symmetric split
        fprintf(' done\n\n');
    else
        fprintf('  Preconditioner: none\n\n');
        M1 = [];
        M2 = [];
    end

    res = struct();
    res.n = n;
    res.grid = m;

    % === MLDIVIDE ===
    tic;
    x_mldiv = A \ B;
    res.t_mldiv = toc;
    res.res_mldiv = max_residual(A, x_mldiv, B);

    % === QMR (point method, each RHS separately, with preconditioner) ===
    [res.t_qmr, res.iter_qmr, res.res_qmr] = run_qmr(A, B, tol, maxiter, M1, M2);

    % === BLQMR with different block sizes (with preconditioner) ===
    res.blqmr = struct('bs', block_sizes, 'time', zeros(size(block_sizes)), ...
                       'iters', zeros(size(block_sizes)), 'res', zeros(size(block_sizes)), ...
                       'flag', zeros(size(block_sizes)), 'batches', zeros(size(block_sizes)));

    for bi = 1:length(block_sizes)
        bs = block_sizes(bi);
        num_full = floor(total_rhs / bs);
        remainder = mod(total_rhs, bs);
        num_batches = num_full + (remainder > 0);

        [res.blqmr.time(bi), res.blqmr.iters(bi), res.blqmr.res(bi), res.blqmr.flag(bi)] = ...
            run_blqmr(A, B, bs, tol, maxiter, M1, M2);
        res.blqmr.batches(bi) = num_batches;
    end

    % === PRINT RESULTS ===
    fprintf('  TIMING & ACCURACY:\n');
    fprintf('  %-12s %10s %10s %8s %12s %8s\n', 'Method', 'Time(s)', 'Iters', 'Batches', 'Residual', 'Flag');
    fprintf('  %-12s %10s %10s %8s %12s %8s\n', '------------', '----------', '----------', '--------', '------------', '--------');
    fprintf('  %-12s %10.3f %10s %8s %12.1e %8s\n', 'mldivide', res.t_mldiv, '--', '--', res.res_mldiv, '--');
    fprintf('  %-12s %10.3f %10d %8s %12.1e %8s\n', 'QMR', res.t_qmr, res.iter_qmr, '--', res.res_qmr, '--');
    for bi = 1:length(block_sizes)
        bs = block_sizes(bi);
        num_full = floor(total_rhs / bs);
        remainder = mod(total_rhs, bs);
        if remainder > 0
            batch_str = sprintf('%d+%d', num_full, 1);
        else
            batch_str = sprintf('%d', num_full);
        end
        fprintf('  %-12s %10.3f %10d %8s %12.1e %8d\n', sprintf('BLQMR-%d', bs), ...
                res.blqmr.time(bi), res.blqmr.iters(bi), batch_str, res.blqmr.res(bi), res.blqmr.flag(bi));
    end

    % === SPEEDUP COMPARISON ===
    fprintf('\n  SPEEDUP (time ratio, >1 means method is faster):\n');
    fprintf('  %-12s %12s %12s %12s\n', 'Method', 'vs mldivide', 'vs QMR', 'vs BLQMR-1');
    fprintf('  %-12s %12s %12s %12s\n', '------------', '------------', '------------', '------------');
    fprintf('  %-12s %12s %12.2fx %12s\n', 'mldivide', '--', res.t_qmr / res.t_mldiv, '--');
    fprintf('  %-12s %12.2fx %12s %12s\n', 'QMR', res.t_mldiv / res.t_qmr, '--', '--');
    for bi = 1:length(block_sizes)
        bs = block_sizes(bi);
        t = res.blqmr.time(bi);
        fprintf('  %-12s %12.2fx %12.2fx %12.2fx\n', sprintf('BLQMR-%d', bs), ...
                res.t_mldiv / t, res.t_qmr / t, res.blqmr.time(1) / t);
    end

    % === ITERATION EFFICIENCY ===
    fprintf('\n  BLQMR ITERATION EFFICIENCY:\n');
    fprintf('  %-12s %10s %12s %12s\n', 'Block Size', 'Iters', 'Ratio', 'Ideal');
    fprintf('  %-12s %10s %12s %12s\n', '------------', '----------', '------------', '------------');
    base_iters = res.blqmr.iters(1);
    for bi = 1:length(block_sizes)
        bs = block_sizes(bi);
        ratio = res.blqmr.iters(bi) / base_iters;
        fprintf('  %-12d %10d %12.3f %12.3f\n', bs, res.blqmr.iters(bi), ratio, 1 / bs);
    end

    all_results{gi} = res;
    all_results{gi}.t_qmr = res.t_qmr;
    all_results{gi}.iter_qmr = res.iter_qmr;
    fprintf('\n');
end

% === FINAL SUMMARY ===
print_summary(all_results, grid_sizes, block_sizes, total_rhs);
end

function [t, iters, residual] = run_qmr(A, B, tol, maxiter, M1, M2)
% RUN_QMR Solve each RHS with MATLAB's QMR (point method) with split preconditioner
nrhs = size(B, 2);
n = size(A, 1);
X = zeros(n, nrhs, 'like', B);
t = 0;
iters = 0;

for k = 1:nrhs
    tic;
    if isempty(M1)
        [X(:, k), ~, ~, iter] = qmr(A, B(:, k), tol, maxiter);
    else
        [X(:, k), ~, ~, iter] = qmr(A, B(:, k), tol, maxiter, M1, M2);
    end
    t = t + toc;
    iters = iters + iter;
end
residual = max_residual(A, X, B);
end

function [t, iters, residual, flag] = run_blqmr(A, B, bs, tol, maxiter, M1, M2)
% RUN_BLQMR Solve with BLQMR using given block size with split preconditioner
%   Handles remainder RHS when nrhs is not divisible by bs
nrhs = size(B, 2);
n = size(A, 1);
X = zeros(n, nrhs, 'like', B);

num_full_batches = floor(nrhs / bs);
remainder = mod(nrhs, bs);

t = 0;
iters = 0;
flag = 0;

% Process full batches
for batch = 1:num_full_batches
    cols = (batch - 1) * bs + (1:bs);
    tic;
    if isempty(M1)
        [X(:, cols), f, ~, it] = blqmr(A, B(:, cols), tol, maxiter);
    else
        [X(:, cols), f, ~, it] = blqmr(A, B(:, cols), tol, maxiter, M1, M2);
    end
    t = t + toc;
    iters = iters + it;
    flag = max(flag, f);
end

% Process remainder batch (if any)
if remainder > 0
    cols = num_full_batches * bs + (1:remainder);
    tic;
    if isempty(M1)
        [X(:, cols), f, ~, it] = blqmr(A, B(:, cols), tol, maxiter);
    else
        [X(:, cols), f, ~, it] = blqmr(A, B(:, cols), tol, maxiter, M1, M2);
    end
    t = t + toc;
    iters = iters + it;
    flag = max(flag, f);
end

residual = max_residual(A, X, B);
end

function print_summary(all_results, grid_sizes, block_sizes, total_rhs)
% PRINT_SUMMARY Print final comparison tables

fprintf('=====================================================================================================\n');
fprintf('SUMMARY: ALL METHODS COMPARISON (%d RHS)\n', total_rhs);
fprintf('=====================================================================================================\n\n');

% Timing table
fprintf('WALL CLOCK TIME (seconds):\n');
fprintf('%8s %10s %10s', 'Grid', 'mldivide', 'QMR');
for bs = block_sizes
    fprintf(' %8s', sprintf('BL-%d', bs));
end
fprintf('\n');
fprintf('%8s %10s %10s', '--------', '----------', '----------');
for bs = block_sizes
    fprintf(' %8s', '--------');
end
fprintf('\n');

for gi = 1:length(grid_sizes)
    r = all_results{gi};
    fprintf('%7d^3 %10.3f %10.3f', r.grid, r.t_mldiv, r.t_qmr);
    for bi = 1:length(block_sizes)
        fprintf(' %8.3f', r.blqmr.time(bi));
    end
    fprintf('\n');
end

% Speedup vs mldivide
fprintf('\nSPEEDUP vs MLDIVIDE (>1 = faster than mldivide):\n');
fprintf('%8s %10s %10s', 'Grid', 'mldivide', 'QMR');
for bs = block_sizes
    fprintf(' %8s', sprintf('BL-%d', bs));
end
fprintf('\n');
fprintf('%8s %10s %10s', '--------', '----------', '----------');
for bs = block_sizes
    fprintf(' %8s', '--------');
end
fprintf('\n');

for gi = 1:length(grid_sizes)
    r = all_results{gi};
    fprintf('%7d^3 %10s %10.2f', r.grid, '1.00', r.t_mldiv / r.t_qmr);
    for bi = 1:length(block_sizes)
        fprintf(' %8.2f', r.t_mldiv / r.blqmr.time(bi));
    end
    fprintf('\n');
end

% Speedup vs QMR
fprintf('\nSPEEDUP vs QMR (>1 = faster than QMR):\n');
fprintf('%8s %10s %10s', 'Grid', 'mldivide', 'QMR');
for bs = block_sizes
    fprintf(' %8s', sprintf('BL-%d', bs));
end
fprintf('\n');
fprintf('%8s %10s %10s', '--------', '----------', '----------');
for bs = block_sizes
    fprintf(' %8s', '--------');
end
fprintf('\n');

for gi = 1:length(grid_sizes)
    r = all_results{gi};
    fprintf('%7d^3 %10.2f %10s', r.grid, r.t_qmr / r.t_mldiv, '1.00');
    for bi = 1:length(block_sizes)
        fprintf(' %8.2f', r.t_qmr / r.blqmr.time(bi));
    end
    fprintf('\n');
end

% Iteration efficiency
fprintf('\nBLQMR ITERATION RATIO (vs block size 1, ideal = 1/block_size):\n');
fprintf('%8s', 'Grid');
for bs = block_sizes
    fprintf(' %8s', sprintf('BL-%d', bs));
end
fprintf('\n');
fprintf('%8s', '--------');
for bs = block_sizes
    fprintf(' %8s', '--------');
end
fprintf('\n');

for gi = 1:length(grid_sizes)
    r = all_results{gi};
    base = r.blqmr.iters(1);
    fprintf('%7d^3', r.grid);
    for bi = 1:length(block_sizes)
        fprintf(' %8.3f', r.blqmr.iters(bi) / base);
    end
    fprintf('\n');
end
fprintf('%8s', 'Ideal');
for bs = block_sizes
    fprintf(' %8.3f', 1 / bs);
end
fprintf('\n');

% Best method per grid
fprintf('\n=====================================================================================================\n');
fprintf('BEST METHOD PER GRID SIZE:\n');
fprintf('=====================================================================================================\n');
for gi = 1:length(grid_sizes)
    r = all_results{gi};
    times = [r.t_mldiv, r.t_qmr, r.blqmr.time];
    names = [{'mldivide', 'QMR'}, arrayfun(@(x) sprintf('BLQMR-%d', x), block_sizes, 'Uni', 0)];
    [best_time, idx] = min(times);
    fprintf('  %d^3: %s (%.3fs) - ', r.grid, names{idx}, best_time);
    fprintf('mldivide=%.3fs, QMR=%.3fs, BLQMR-64=%.3fs\n', r.t_mldiv, r.t_qmr, r.blqmr.time(end));
end

% Iteration comparison
fprintf('\n=====================================================================================================\n');
fprintf('ITERATION COUNT COMPARISON (total iterations for all %d RHS):\n', total_rhs);
fprintf('=====================================================================================================\n');
fprintf('%8s %10s %10s %10s\n', 'Grid', 'QMR', 'BLQMR-1', 'BLQMR-64');
fprintf('%8s %10s %10s %10s\n', '--------', '----------', '----------', '----------');
for gi = 1:length(grid_sizes)
    r = all_results{gi};
    fprintf('%7d^3 %10d %10d %10d\n', r.grid, r.iter_qmr, r.blqmr.iters(1), r.blqmr.iters(end));
end
fprintf('\nNote: QMR and BLQMR-1 should have similar iteration counts (both are point methods).\n');
fprintf('      BLQMR-64 shows the block acceleration effect.\n');
fprintf('=====================================================================================================\n');
end

function A = assemble_helmholtz_fem(node, elem)
% ASSEMBLE_HELMHOLTZ_FEM Complex symmetric Helmholtz-like FEM matrix
n = size(node, 1);
nelem = size(elem, 1);

II = zeros(16 * nelem, 1);
JJ = zeros(16 * nelem, 1);
VV = zeros(16 * nelem, 1, 'like', 1i);
cnt = 0;

for e = 1:nelem
    idx = elem(e, :);
    coords = node(idx, :);
    d1 = coords(2, :) - coords(1, :);
    d2 = coords(3, :) - coords(1, :);
    d3 = coords(4, :) - coords(1, :);
    J = [d1; d2; d3]';
    vol = abs(det(J)) / 6;
    if vol < 1e-15
        continue
    end

    invJ = inv(J);
    grad_ref = [-1 -1 -1; 1 0 0; 0 1 0; 0 0 1]';
    grad_N = invJ' * grad_ref;
    Ke = vol * (grad_N' * grad_N);
    Me = vol / 20 * (ones(4) + eye(4));

    % Helmholtz with absorption (complex symmetric)
    Ae = Ke - 1.0 * Me + 0.3i * Me;

    for i = 1:4
        for j = 1:4
            cnt = cnt + 1;
            II(cnt) = idx(i);
            JJ(cnt) = idx(j);
            VV(cnt) = Ae(i, j);
        end
    end
end

II = II(1:cnt);
JJ = JJ(1:cnt);
VV = VV(1:cnt);
A = sparse(II, JJ, VV, n, n);
A = (A + A.') / 2;
A = A + 0.01 * mean(abs(diag(A))) * speye(n);
end

function B = create_distributed_sources(node, elem, nrhs)
% CREATE_DISTRIBUTED_SOURCES Spatially distributed point sources
n = size(node, 1);
B = zeros(n, nrhs, 'like', 1i);

xr = [min(node(:, 1)), max(node(:, 1))];
yr = [min(node(:, 2)), max(node(:, 2))];
zr = [min(node(:, 3)), max(node(:, 3))];

ns = ceil(nrhs^(1 / 3));
cnt = 0;
src_pos = zeros(nrhs, 3);

for iz = 1:ns
    for iy = 1:ns
        for ix = 1:ns
            cnt = cnt + 1;
            if cnt > nrhs
                break
            end
            fx = 0.15 + 0.7 * (ix - 0.5) / ns;
            fy = 0.15 + 0.7 * (iy - 0.5) / ns;
            fz = 0.15 + 0.7 * (iz - 0.5) / ns;
            src_pos(cnt, :) = [xr(1) + fx * diff(xr), yr(1) + fy * diff(yr), zr(1) + fz * diff(zr)];
        end
        if cnt > nrhs
            break
        end
    end
    if cnt > nrhs
        break
    end
end

[eidx, bary] = tsearchn(node, elem, src_pos);
for k = 1:nrhs
    phase = exp(1i * 2 * pi * (k - 1) / nrhs);
    if ~isnan(eidx(k))
        B(elem(eidx(k), :), k) = phase * bary(k, :)';
    else
        [~, nearest] = min(sum((node - src_pos(k, :)).^2, 2));
        B(nearest, k) = phase;
    end
end
end

function r = max_residual(A, X, B)
r = 0;
for i = 1:size(B, 2)
    bn = norm(B(:, i));
    if bn > 0
        r = max(r, norm(A * X(:, i) - B(:, i)) / bn);
    end
end
end
