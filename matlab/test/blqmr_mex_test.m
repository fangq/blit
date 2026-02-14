% BLQMR_TEST  Regression test for BLQMR solver (MATLAB + optional Fortran MEX).

fprintf('BLIT BLQMR Test\n');
fprintf('========================================\n');

has_mex = exist('blqmr_', 'file') == 3;
if has_mex
    fprintf('Fortran MEX (blqmr_) detected - testing both backends\n');
else
    fprintf('Fortran MEX not found - testing MATLAB backend only\n');
end

%% Build test matrix (same as Fortran self-test)
n = 5;
Ap = [0, 2, 5, 9, 10, 12] + 1;  % 1-based for MATLAB sparse()
Ai = [0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4] + 1;
Ax = [2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1.];
% Reconstruct sparse matrix from CSC
rows = Ai;
cols = zeros(length(Ai), 1);
for j = 1:n
    cols(Ap(j):Ap(j + 1) - 1) = j;
end
A = sparse(rows, cols, Ax, n, n);

b1 = [8.0; 45.0; -3.0; 3.0; 19.0];
b2 = [18.0; 45.0; -3.0; 3.0; 19.0];
B = [b1, b2];

%% Test 1: Single RHS, no preconditioner
fprintf('\nTest 1: Single RHS, no preconditioning (MATLAB)\n');
opt1 = struct('usefortran', 0);
[x, flag, relres, iter] = blqmr(A, b1, 1e-5, 100, [], [], [], opt1);
res_norm = norm(A * x - b1);
fprintf('  flag=%d, iter=%d, relres=%.2e, ||Ax-b||=%.2e\n', flag, iter, relres, res_norm);
assert(res_norm < 1e-6, 'Test 1 FAILED');
fprintf('  PASSED\n');

%% Test 2: Single RHS with opt.precond='diag'
fprintf('\nTest 2: Single RHS, diagonal preconditioner (MATLAB)\n');
opt2 = struct('precond', 'diag', 'usefortran', 0);
[x, flag, relres, iter] = blqmr(A, b1, 1e-5, 100, [], [], [], opt2);
res_norm = norm(A * x - b1);
fprintf('  flag=%d, iter=%d, relres=%.2e, ||Ax-b||=%.2e\n', flag, iter, relres, res_norm);
assert(res_norm < 1e-6, 'Test 2 FAILED');
fprintf('  PASSED\n');

%% Test 3: Multiple RHS
fprintf('\nTest 3: Multiple RHS (MATLAB)\n');
opt3 = struct('usefortran', 0);
[X, flag, relres, iter] = blqmr(A, B, 1e-5, 100, [], [], [], opt3);
res_norm = norm(A * X - B, 'fro');
fprintf('  flag=%d, iter=%d, relres=%.2e, ||AX-B||_F=%.2e\n', flag, iter, relres, res_norm);
assert(res_norm < 1e-4, 'Test 3 FAILED');
fprintf('  PASSED\n');

%% Test 4: Blocksize=1 batching
fprintf('\nTest 4: Blocksize=1 batching (MATLAB)\n');
opt4 = struct('blocksize', 1, 'usefortran', 0);
[X, flag, relres, iter] = blqmr(A, B, 1e-5, 100, [], [], [], opt4);
res_norm = norm(A * X - B, 'fro');
fprintf('  flag=%d, iter=%d, relres=%.2e, ||AX-B||_F=%.2e\n', flag, iter, relres, res_norm);
assert(res_norm < 1e-4, 'Test 4 FAILED');
fprintf('  PASSED\n');

%% Test 5-8: Fortran MEX tests (only if blqmr_ exists)
if has_mex
    fprintf('\n---- Fortran MEX backend tests ----\n');

    %% Test 5: Single RHS via MEX (auto-dispatch)
    fprintf('\nTest 5: Single RHS via Fortran MEX (ILU-left, pcond=1)\n');
    opt5 = struct('precond', 'ilu');
    [x, flag, relres, iter] = blqmr(A, b1, 1e-5, 100, [], [], [], opt5);
    res_norm = norm(A * x - b1);
    fprintf('  flag=%d, iter=%d, relres=%.2e, ||Ax-b||=%.2e\n', flag, iter, relres, res_norm);
    assert(res_norm < 1e-6, 'Test 5 FAILED');
    fprintf('  PASSED\n');

    %% Test 6: Single RHS, no preconditioning via MEX
    fprintf('\nTest 6: Single RHS via Fortran MEX (no precond)\n');
    [x, flag, relres, iter] = blqmr(A, b1, 1e-5, 100);
    res_norm = norm(A * x - b1);
    fprintf('  flag=%d, iter=%d, relres=%.2e, ||Ax-b||=%.2e\n', flag, iter, relres, res_norm);
    assert(res_norm < 1e-4, 'Test 6 FAILED');
    fprintf('  PASSED\n');

    %% Test 7: Multiple RHS via MEX
    fprintf('\nTest 7: Multiple RHS via Fortran MEX\n');
    opt7 = struct('precond', 'ilu');
    [X, flag, relres, iter] = blqmr(A, B, 1e-5, 100, [], [], [], opt7);
    res_norm = norm(A * X - B, 'fro');
    fprintf('  flag=%d, iter=%d, relres=%.2e, ||AX-B||_F=%.2e\n', flag, iter, relres, res_norm);
    assert(res_norm < 1e-4, 'Test 7 FAILED');
    fprintf('  PASSED\n');

    %% Test 8: Multiple RHS with OpenMP (nblock=1)
    fprintf('\nTest 8: Multiple RHS via Fortran MEX + OpenMP (nblock=1)\n');
    opt8 = struct('precond', 'ilu', 'blocksize', 1);
    [X, flag, relres, iter] = blqmr(A, B, 1e-5, 100, [], [], [], opt8);
    res_norm = norm(A * X - B, 'fro');
    fprintf('  flag=%d, iter=%d, relres=%.2e, ||AX-B||_F=%.2e\n', flag, iter, relres, res_norm);
    assert(res_norm < 1e-4, 'Test 8 FAILED');
    fprintf('  PASSED\n');

    %% Test 9: Direct blqmr_ call
    fprintf('\nTest 9: Direct blqmr_() call\n');
    [x, flag, relres, iter] = blqmr_(A, b1, 1e-5, 100, 1, 0.001, 0);
    res_norm = norm(A * x - b1);
    fprintf('  flag=%d, iter=%d, relres=%.2e, ||Ax-b||=%.2e\n', flag, iter, relres, res_norm);
    assert(res_norm < 1e-6, 'Test 9 FAILED');
    fprintf('  PASSED\n');
end

%% Test 10: Larger random SPD system
fprintf('\nTest 10: Random 100x100 SPD system\n');
rng(42);
n2 = 100;
T = sprandn(n2, n2, 0.05);
A2 = T' * T + 10 * speye(n2);
b2 = randn(n2, 1);

opt10 = struct('precond', 'diag');
[x2, flag2, relres2, iter2] = blqmr(A2, b2, 1e-10, 200, [], [], [], opt10);
res_norm = norm(A2 * x2 - b2);
fprintf('  flag=%d, iter=%d, relres=%.2e, ||Ax-b||=%.2e\n', flag2, iter2, relres2, res_norm);
assert(flag2 == 0, 'Test 10 FAILED: did not converge');
fprintf('  PASSED\n');

fprintf('\n========================================\n');
fprintf('All tests PASSED.\n');
