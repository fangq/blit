%% Benchmark sweep: Grid size vs RHS count
% Find crossover points where BLQMR beats direct solver.
%
% Uses block size of 4 for BLQMR.
% Complex symmetric Helmholtz FEM matrix with split Jacobi preconditioner.
% Matches Python benchmark configuration.

clear; clc;

%% Configuration
grid_sizes = [10, 15, 20, 25, 30];  % Grid dimensions (n = gridsize^3)
rhs_counts = [1, 2, 4, 8, 16, 32, 64, 128];
block_size = 4;
tol = 1e-6;
maxiter = 2000;
n_runs = 2;  % Number of runs (take minimum time)

fprintf('================================================================================\n');
fprintf('BENCHMARK SWEEP: Grid Size vs RHS Count\n');
fprintf('Finding crossover points where BLQMR beats direct solver\n');
fprintf('================================================================================\n\n');

fprintf('Configuration:\n');
fprintf('  BLQMR block size: %d\n', block_size);
fprintf('  Tolerance: %.0e\n', tol);
fprintf('  Max iterations: %d\n', maxiter);
fprintf('  Preconditioner: split Jacobi (M1 = M2 = sqrt(D))\n');
fprintf('  Matrix: complex symmetric Helmholtz FEM (tetrahedral mesh)\n');
fprintf('  Grid sizes: %s\n', mat2str(grid_sizes));
fprintf('  RHS counts: %s\n', mat2str(rhs_counts));
fprintf('\n');

%% Initialize results storage
results = struct();
results.grid_sizes = grid_sizes;
results.rhs_counts = rhs_counts;
results.t_direct = zeros(length(grid_sizes), length(rhs_counts));
results.t_blqmr = zeros(length(grid_sizes), length(rhs_counts));
results.iter_blqmr = zeros(length(grid_sizes), length(rhs_counts));
results.flag_blqmr = zeros(length(grid_sizes), length(rhs_counts));
results.res_direct = zeros(length(grid_sizes), length(rhs_counts));
results.res_blqmr = zeros(length(grid_sizes), length(rhs_counts));
results.n = zeros(length(grid_sizes), 1);
results.nnz = zeros(length(grid_sizes), 1);

%% Run benchmarks
for gi = 1:length(grid_sizes)
    gridsize = grid_sizes(gi);
    
    fprintf('================================================================================\n');
    fprintf('GRID %d³\n', gridsize);
    fprintf('================================================================================\n');
    
    % Create mesh and matrix
    [node, elem] = meshgrid6(0:gridsize-1, 0:gridsize-1, 0:gridsize-1);
    n = size(node, 1);
    A = assemble_helmholtz_fem(node, elem);
    
    results.n(gi) = n;
    results.nnz(gi) = nnz(A);
    
    fprintf('  Matrix: n=%d, nnz=%d, complex symmetric\n', n, nnz(A));
    
    % Create split Jacobi preconditioner
    [M1, M2] = create_split_jacobi_precond(A);
    
    for ri = 1:length(rhs_counts)
        nrhs = rhs_counts(ri);
        fprintf('\n  RHS=%d:\n', nrhs);
        
        % Generate RHS (distributed sources)
        rng(42);  % For reproducibility
        B = create_distributed_sources(node, elem, nrhs);
        
        %% Direct solver (mldivide)
        fprintf('    Direct (mldivide)...');
        times_direct = zeros(n_runs, 1);
        for run = 1:n_runs
            tic;
            X_direct = A \ B;
            times_direct(run) = toc;
        end
        t_direct = min(times_direct);
        res_direct = max_residual(A, X_direct, B);
        fprintf(' %.3fs (res=%.2e)\n', t_direct, res_direct);
        
        results.t_direct(gi, ri) = t_direct;
        results.res_direct(gi, ri) = res_direct;
        
        %% BLQMR with block size 4
        fprintf('    BLQMR-%d...', block_size);
        times_blqmr = zeros(n_runs, 1);
        for run = 1:n_runs
            tic;
            [X_blqmr, total_iters, max_flag] = solve_blqmr_batched(A, B, M1, M2, block_size, tol, maxiter);
            times_blqmr(run) = toc;
        end
        t_blqmr = min(times_blqmr);
        
        if max_flag <= 1
            res_blqmr = max_residual(A, X_blqmr, B);
            fprintf(' %.3fs (%d iters, res=%.2e)\n', t_blqmr, total_iters, res_blqmr);
        else
            res_blqmr = inf;
            fprintf(' FAILED (flag=%d)\n', max_flag);
        end
        
        results.t_blqmr(gi, ri) = t_blqmr;
        results.iter_blqmr(gi, ri) = total_iters;
        results.flag_blqmr(gi, ri) = max_flag;
        results.res_blqmr(gi, ri) = res_blqmr;
    end
end

%% Print summary tables
print_summary(results);

%% Plot results
plot_results(results);

%% Save results
save('benchmark_sweep_results.mat', 'results');
fprintf('\nResults saved to benchmark_sweep_results.mat\n');


%% ========================================================================
%  Helper Functions
%% ========================================================================

function [node, elem] = meshgrid6(x, y, z)
    % Generate tetrahedral mesh from regular gridsize (6 tets per cube)
    % Matches Python meshgrid6 function
    
    nx = length(x);
    ny = length(y);
    nz = length(z);
    
    % Create nodes
    [X, Y, Z] = ndgrid(x, y, z);
    node = [X(:), Y(:), Z(:)];
    
    % Create elements (6 tets per cube)
    elem = [];
    
    idx = @(i, j, k) (i-1)*ny*nz + (j-1)*nz + k;
    
    for i = 1:nx-1
        for j = 1:ny-1
            for k = 1:nz-1
                % 8 corners of the cube (1-based indexing)
                n0 = idx(i, j, k);
                n1 = idx(i+1, j, k);
                n2 = idx(i+1, j+1, k);
                n3 = idx(i, j+1, k);
                n4 = idx(i, j, k+1);
                n5 = idx(i+1, j, k+1);
                n6 = idx(i+1, j+1, k+1);
                n7 = idx(i, j+1, k+1);
                
                % 6 tetrahedra
                elem = [elem;
                    n0, n1, n3, n4;
                    n1, n2, n3, n6;
                    n1, n4, n5, n6;
                    n3, n4, n6, n7;
                    n1, n3, n4, n6;
                    n1, n2, n6, n5];
            end
        end
    end
end

function A = assemble_helmholtz_fem(node, elem)
    % Assemble complex symmetric Helmholtz-like FEM matrix
    % A = K - 1.0*M + 0.3i*M + regularization
    
    n = size(node, 1);
    nelem = size(elem, 1);
    
    % Preallocate COO arrays
    max_entries = 16 * nelem;
    II = zeros(max_entries, 1);
    JJ = zeros(max_entries, 1);
    VV = zeros(max_entries, 1);
    cnt = 0;
    
    for e = 1:nelem
        idx = elem(e, :);
        coords = node(idx, :);
        
        % Jacobian
        d1 = coords(2, :) - coords(1, :);
        d2 = coords(3, :) - coords(1, :);
        d3 = coords(4, :) - coords(1, :);
        J = [d1; d2; d3]';
        
        vol = abs(det(J)) / 6.0;
        if vol < 1e-15
            continue;
        end
        
        % Gradient of shape functions
        invJ = inv(J);
        grad_ref = [-1, -1, -1; 1, 0, 0; 0, 1, 0; 0, 0, 1]';
        grad_N = invJ' * grad_ref;
        
        % Element stiffness and mass
        Ke = vol * (grad_N' * grad_N);
        Me = vol / 20.0 * (ones(4, 4) + eye(4));
        
        % Helmholtz with absorption (complex symmetric)
        Ae = Ke - 1.0 * Me + 0.3i * Me;
        
        % Assemble
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
    
    % Symmetrize and add regularization
    A = (A + A.') / 2;
    A = A + 0.01 * mean(abs(diag(A))) * speye(n);
end

function B = create_distributed_sources(node, elem, nrhs)
    % Create spatially distributed point sources
    
    n = size(node, 1);
    B = zeros(n, nrhs);
    
    xr = [min(node(:,1)), max(node(:,1))];
    yr = [min(node(:,2)), max(node(:,2))];
    zr = [min(node(:,3)), max(node(:,3))];
    
    % Generate source positions
    ns = max(1, ceil(nrhs^(1/3)));
    src_pos = [];
    
    for iz = 1:ns
        for iy = 1:ns
            for ix = 1:ns
                if size(src_pos, 1) >= nrhs
                    break;
                end
                fx = 0.15 + 0.7 * (ix - 0.5) / ns;
                fy = 0.15 + 0.7 * (iy - 0.5) / ns;
                fz = 0.15 + 0.7 * (iz - 0.5) / ns;
                pos = [xr(1) + fx * (xr(2) - xr(1)), ...
                       yr(1) + fy * (yr(2) - yr(1)), ...
                       zr(1) + fz * (zr(2) - zr(1))];
                src_pos = [src_pos; pos];
            end
        end
    end
    src_pos = src_pos(1:nrhs, :);
    
    % Find nearest nodes
    for k = 1:nrhs
        phase = exp(1i * 2 * pi * (k-1) / max(nrhs, 1));
        dists = sum((node - src_pos(k, :)).^2, 2);
        [~, nearest] = min(dists);
        B(nearest, k) = phase;
    end
end

function [M1, M2] = create_split_jacobi_precond(A)
    % Create split Jacobi preconditioner: M1 = M2 = sqrt(D)
    
    d = diag(A);
    small_thresh = max(max(abs(d)) * 1e-14, 1e-14);
    d(abs(d) < small_thresh) = 1.0;
    sqrt_d = sqrt(d);
    
    M1 = spdiags(sqrt_d, 0, length(d), length(d));
    M2 = M1;
end

function r = max_residual(A, X, B)
    % Compute maximum relative residual across all RHS
    nrhs = size(B, 2);
    r = 0;
    for i = 1:nrhs
        bn = norm(B(:, i));
        if bn > 0
            r = max(r, norm(A * X(:, i) - B(:, i)) / bn);
        end
    end
end

function [X, total_iters, max_flag] = solve_blqmr_batched(A, B, M1, M2, block_size, tol, maxiter)
    % Solve with BLQMR using batched processing
    
    n = size(A, 1);
    nrhs = size(B, 2);
    X = zeros(n, nrhs);
    
    n_full_batches = floor(nrhs / block_size);
    remainder = mod(nrhs, block_size);
    
    total_iters = 0;
    max_flag = 0;
    
    % Process full batches
    for batch = 1:n_full_batches
        start_idx = (batch - 1) * block_size + 1;
        end_idx = start_idx + block_size - 1;
        B_batch = B(:, start_idx:end_idx);
        
        [x_batch, flag, ~, iter] = blqmr(A, B_batch, tol, maxiter, M1, M2);
        
        X(:, start_idx:end_idx) = x_batch;
        total_iters = total_iters + iter;
        max_flag = max(max_flag, flag);
    end
    
    % Process remainder
    if remainder > 0
        start_idx = n_full_batches * block_size + 1;
        B_rem = B(:, start_idx:end);
        
        [x_rem, flag, ~, iter] = blqmr(A, B_rem, tol, maxiter, M1, M2);
        
        X(:, start_idx:end) = x_rem;
        total_iters = total_iters + iter;
        max_flag = max(max_flag, flag);
    end
end

function print_summary(results)
    % Print summary tables
    
    fprintf('\n');
    fprintf('========================================================================================================================\n');
    fprintf('SUMMARY: Direct (mldivide) vs BLQMR-4\n');
    fprintf('========================================================================================================================\n');
    
    % Direct solver times
    fprintf('\nDirect (mldivide) times (seconds):\n');
    fprintf('────────────────────────────────────────────────────────────────────────────────────────────────────\n');
    fprintf('%12s │', 'Grid');
    for ri = 1:length(results.rhs_counts)
        fprintf(' %7d │', results.rhs_counts(ri));
    end
    fprintf('\n');
    fprintf('────────────────────────────────────────────────────────────────────────────────────────────────────\n');
    
    for gi = 1:length(results.grid_sizes)
        gridsize = results.grid_sizes(gi);
        fprintf('%2d³ (n=%5d) │', gridsize, results.n(gi));
        for ri = 1:length(results.rhs_counts)
            fprintf(' %7.3f │', results.t_direct(gi, ri));
        end
        fprintf('\n');
    end
    
    % BLQMR times
    fprintf('\nBLQMR-4 times (seconds):\n');
    fprintf('────────────────────────────────────────────────────────────────────────────────────────────────────\n');
    fprintf('%12s │', 'Grid');
    for ri = 1:length(results.rhs_counts)
        fprintf(' %7d │', results.rhs_counts(ri));
    end
    fprintf('\n');
    fprintf('────────────────────────────────────────────────────────────────────────────────────────────────────\n');
    
    for gi = 1:length(results.grid_sizes)
        gridsize = results.grid_sizes(gi);
        fprintf('%2d³ (n=%5d) │', gridsize, results.n(gi));
        for ri = 1:length(results.rhs_counts)
            if results.flag_blqmr(gi, ri) <= 1
                fprintf(' %7.3f │', results.t_blqmr(gi, ri));
            else
                fprintf(' %7s │', 'FAIL');
            end
        end
        fprintf('\n');
    end
    
    % Speedup table
    fprintf('\n');
    fprintf('========================================================================================================================\n');
    fprintf('SPEEDUP: Direct_time / BLQMR_time (>1.0 = BLQMR wins, marked with *)\n');
    fprintf('========================================================================================================================\n');
    fprintf('────────────────────────────────────────────────────────────────────────────────────────────────────\n');
    fprintf('%12s │', 'Grid');
    for ri = 1:length(results.rhs_counts)
        fprintf(' %7d │', results.rhs_counts(ri));
    end
    fprintf('\n');
    fprintf('────────────────────────────────────────────────────────────────────────────────────────────────────\n');
    
    for gi = 1:length(results.grid_sizes)
        gridsize = results.grid_sizes(gi);
        fprintf('%2d³ (n=%5d) │', gridsize, results.n(gi));
        for ri = 1:length(results.rhs_counts)
            if results.flag_blqmr(gi, ri) <= 1 && results.t_blqmr(gi, ri) > 0
                speedup = results.t_direct(gi, ri) / results.t_blqmr(gi, ri);
                if speedup >= 1.0
                    fprintf(' *%5.2f* │', speedup);
                else
                    fprintf('  %5.2f  │', speedup);
                end
            else
                fprintf(' %7s │', 'N/A');
            end
        end
        fprintf('\n');
    end
    
    % Crossover analysis
    fprintf('\n');
    fprintf('========================================================================================================================\n');
    fprintf('CROSSOVER ANALYSIS\n');
    fprintf('========================================================================================================================\n');
    fprintf('\nFor each gridsize size, find minimum RHS where BLQMR becomes faster than Direct:\n\n');
    
    for gi = 1:length(results.grid_sizes)
        gridsize = results.grid_sizes(gi);
        n = results.n(gi);
        crossover_rhs = [];
        crossover_speedup = [];
        
        for ri = 1:length(results.rhs_counts)
            nrhs = results.rhs_counts(ri);
            if results.flag_blqmr(gi, ri) <= 1
                t_direct = results.t_direct(gi, ri);
                t_blqmr = results.t_blqmr(gi, ri);
                if t_direct > t_blqmr
                    crossover_rhs = nrhs;
                    crossover_speedup = t_direct / t_blqmr;
                    break;
                end
            end
        end
        
        if ~isempty(crossover_rhs)
            fprintf('  Grid %2d³ (n=%6d): BLQMR wins at RHS >= %3d  (speedup = %.2fx)\n', ...
                    gridsize, n, crossover_rhs, crossover_speedup);
        else
            % Find best speedup
            best_speedup = 0;
            best_rhs = [];
            for ri = 1:length(results.rhs_counts)
                if results.flag_blqmr(gi, ri) <= 1 && results.t_blqmr(gi, ri) > 0
                    speedup = results.t_direct(gi, ri) / results.t_blqmr(gi, ri);
                    if speedup > best_speedup
                        best_speedup = speedup;
                        best_rhs = results.rhs_counts(ri);
                    end
                end
            end
            if best_speedup > 0
                fprintf('  Grid %2d³ (n=%6d): Direct wins all tested RHS (best BLQMR ratio: %.2fx at RHS=%d)\n', ...
                        gridsize, n, best_speedup, best_rhs);
            else
                fprintf('  Grid %2d³ (n=%6d): No valid BLQMR results\n', gridsize, n);
            end
        end
    end
end

function plot_results(results)
    % Create visualization plots
    
    figure('Position', [100, 100, 1200, 500]);
    
    colors = lines(length(results.grid_sizes));
    
    %% Time comparison plot
    subplot(1, 2, 1);
    hold on;
    
    legends = {};
    for gi = 1:length(results.grid_sizes)
        gridsize = results.grid_sizes(gi);
        
        % Direct solver (solid line)
        loglog(results.rhs_counts, results.t_direct(gi, :), '-o', ...
               'Color', colors(gi, :), 'LineWidth', 2, 'MarkerSize', 6);
        legends{end+1} = sprintf('Direct %d³', gridsize);
        
        % BLQMR (dashed line)
        loglog(results.rhs_counts, results.t_blqmr(gi, :), '--s', ...
               'Color', colors(gi, :), 'LineWidth', 2, 'MarkerSize', 6);
        legends{end+1} = sprintf('BLQMR-4 %d³', gridsize);
    end
    
    xlabel('Number of RHS');
    ylabel('Time (seconds)');
    title('Time Comparison: Direct vs BLQMR-4');
    legend(legends, 'Location', 'northwest', 'NumColumns', 2);
    grid on;
    set(gca, 'XScale', 'log', 'YScale', 'log');
    xlim([1, 128]);
    
    %% Speedup plot
    subplot(1, 2, 2);
    hold on;
    
    % Reference line at y=1
    plot([1, 128], [1, 1], 'k--', 'LineWidth', 1);
    
    legends = {'Crossover (1.0)'};
    for gi = 1:length(results.grid_sizes)
        gridsize = results.grid_sizes(gi);
        speedup = results.t_direct(gi, :) ./ results.t_blqmr(gi, :);
        
        loglog(results.rhs_counts, speedup, '-o', ...
               'Color', colors(gi, :), 'LineWidth', 2, 'MarkerSize', 6);
        legends{end+1} = sprintf('Grid %d³', gridsize);
    end
    
    xlabel('Number of RHS');
    ylabel('Speedup (Direct / BLQMR)');
    title('Speedup: Values > 1 mean BLQMR is faster');
    legend(legends, 'Location', 'northeast');
    grid on;
    set(gca, 'XScale', 'log', 'YScale', 'log');
    xlim([1, 128]);
    ylim([0.01, 100]);
    
    % Save figure
    saveas(gcf, 'benchmark_sweep_results.png');
    fprintf('\nFigure saved to benchmark_sweep_results.png\n');
end