"""
Benchmark script for BLQMR block size performance.

Compares TOTAL TIME to solve all RHS for:
1. SuperLU (factorize once, solve all)
2. Sequential single-RHS BLQMR (block_size=1, baseline)
3. Block BLQMR with various block sizes (2, 8, 16, 64)

Block sizes: 1, 2, 8, 16, 64
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu
import time
import sys
import os
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from blocksolver import BLQMR_EXT, HAS_NUMBA, get_backend_info
from blocksolver.blqmr import _blqmr_python_impl, make_preconditioner

if BLQMR_EXT:
    from blocksolver.blqmr import _blqmr_fortran


def create_3d_fem_matrix(n_target, matrix_type="real"):
    """
    Create 3D FEM-like sparse symmetric matrix (7-point stencil Laplacian).
    """
    m = max(2, int(round(n_target ** (1 / 3))))
    n = m * m * m

    dtype = np.complex128 if matrix_type == "complex" else np.float64

    diag_main = 6.0 + 0.2j if matrix_type == "complex" else 6.0
    diag_off = -1.0 + 0.1j if matrix_type == "complex" else -1.0

    row, col, data = [], [], []

    for i in range(m):
        for j in range(m):
            for k in range(m):
                idx = i * m * m + j * m + k

                row.append(idx)
                col.append(idx)
                data.append(diag_main)

                if i > 0:
                    row.append(idx)
                    col.append(idx - m * m)
                    data.append(diag_off)
                if i < m - 1:
                    row.append(idx)
                    col.append(idx + m * m)
                    data.append(diag_off)
                if j > 0:
                    row.append(idx)
                    col.append(idx - m)
                    data.append(diag_off)
                if j < m - 1:
                    row.append(idx)
                    col.append(idx + m)
                    data.append(diag_off)
                if k > 0:
                    row.append(idx)
                    col.append(idx - 1)
                    data.append(diag_off)
                if k < m - 1:
                    row.append(idx)
                    col.append(idx + 1)
                    data.append(diag_off)

    A = sparse.coo_matrix((data, (row, col)), shape=(n, n), dtype=dtype).tocsc()

    sym_err = sparse.linalg.norm(A - A.T)
    assert sym_err < 1e-14, f"Matrix not symmetric! Error: {sym_err}"

    return A, n, m


def create_rhs(n, nrhs, matrix_type="real", seed=42):
    """Create random right-hand side vectors."""
    np.random.seed(seed)
    if matrix_type == "complex":
        B = np.random.randn(n, nrhs) + 1j * np.random.randn(n, nrhs)
    else:
        B = np.random.randn(n, nrhs)
    return B


def create_localized_rhs(n, grid_m, nrhs, nnz_per_rhs=6, matrix_type="real", seed=42):
    """
    Create RHS vectors with localized non-zeros (like point sources in FEM).

    Each RHS has 4-8 non-zero values at neighboring grid positions,
    with values summing to 1 (normalized source).

    Parameters
    ----------
    n : int
        Total number of unknowns (grid_m^3)
    grid_m : int
        Grid size in each dimension
    nrhs : int
        Number of right-hand side vectors
    nnz_per_rhs : int
        Target number of non-zeros per RHS (4-8 range)
    matrix_type : str
        'real' or 'complex'
    seed : int
        Random seed for reproducibility

    Returns
    -------
    B : ndarray
        RHS matrix (n x nrhs) with localized sources
    source_info : list
        List of (center_idx, center_ijk) for each source
    """
    np.random.seed(seed)
    dtype = np.complex128 if matrix_type == "complex" else np.float64

    B = np.zeros((n, nrhs), dtype=dtype)
    source_info = []

    def ijk_to_idx(i, j, k):
        """Convert (i,j,k) grid coordinates to linear index."""
        return i * grid_m * grid_m + j * grid_m + k

    def idx_to_ijk(idx):
        """Convert linear index to (i,j,k) grid coordinates."""
        i = idx // (grid_m * grid_m)
        j = (idx % (grid_m * grid_m)) // grid_m
        k = idx % grid_m
        return i, j, k

    def get_neighbors(i, j, k, include_center=True):
        """Get neighboring grid points (6-connectivity + center)."""
        neighbors = []
        if include_center:
            neighbors.append((i, j, k))
        # 6-connected neighbors (face neighbors)
        for di, dj, dk in [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < grid_m and 0 <= nj < grid_m and 0 <= nk < grid_m:
                neighbors.append((ni, nj, nk))
        return neighbors

    for rhs_idx in range(nrhs):
        # Pick a random center point (avoid boundaries for full neighborhood)
        margin = 2
        ci = np.random.randint(margin, grid_m - margin)
        cj = np.random.randint(margin, grid_m - margin)
        ck = np.random.randint(margin, grid_m - margin)

        center_idx = ijk_to_idx(ci, cj, ck)
        source_info.append((center_idx, (ci, cj, ck)))

        # Get neighbors
        neighbors = get_neighbors(ci, cj, ck, include_center=True)

        # Randomly select 4-8 points from neighbors
        actual_nnz = min(
            len(neighbors), np.random.randint(4, min(9, len(neighbors) + 1))
        )
        selected = np.random.choice(len(neighbors), actual_nnz, replace=False)

        # Assign random positive weights, normalized to sum to 1
        if matrix_type == "complex":
            weights = np.random.rand(actual_nnz) + 1j * np.random.rand(actual_nnz) * 0.1
        else:
            weights = np.random.rand(actual_nnz)
        weights = weights / np.sum(weights)  # Normalize to sum=1

        # Fill in the RHS vector
        for w_idx, neighbor_idx in enumerate(selected):
            ni, nj, nk = neighbors[neighbor_idx]
            linear_idx = ijk_to_idx(ni, nj, nk)
            B[linear_idx, rhs_idx] = weights[w_idx]

    return B, source_info


def create_clustered_rhs(
    n, grid_m, nrhs, nnz_per_rhs=6, n_clusters=8, matrix_type="real", seed=42
):
    """
    Create RHS vectors with sources clustered in spatial regions.

    Sources within the same cluster are spatially close, which should
    help the block Krylov method find a shared subspace more efficiently.

    Parameters
    ----------
    n : int
        Total number of unknowns
    grid_m : int
        Grid size in each dimension
    nrhs : int
        Number of right-hand side vectors
    nnz_per_rhs : int
        Target non-zeros per RHS
    n_clusters : int
        Number of spatial clusters (RHS are distributed among clusters)
    matrix_type : str
        'real' or 'complex'
    seed : int
        Random seed

    Returns
    -------
    B : ndarray
        RHS matrix with clustered sources
    cluster_info : list
        List of (cluster_id, center_ijk) for each RHS
    """
    np.random.seed(seed)
    dtype = np.complex128 if matrix_type == "complex" else np.float64

    B = np.zeros((n, nrhs), dtype=dtype)
    cluster_info = []

    def ijk_to_idx(i, j, k):
        return i * grid_m * grid_m + j * grid_m + k

    def get_neighbors(i, j, k):
        neighbors = [(i, j, k)]
        for di, dj, dk in [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]:
            ni, nj, nk = i + di, j + dj, k + dk
            if 0 <= ni < grid_m and 0 <= nj < grid_m and 0 <= nk < grid_m:
                neighbors.append((ni, nj, nk))
        return neighbors

    # Define cluster centers spread across the grid
    margin = 3
    cluster_centers = []
    for _ in range(n_clusters):
        ci = np.random.randint(margin, grid_m - margin)
        cj = np.random.randint(margin, grid_m - margin)
        ck = np.random.randint(margin, grid_m - margin)
        cluster_centers.append((ci, cj, ck))

    # Assign each RHS to a cluster
    rhs_per_cluster = nrhs // n_clusters

    for rhs_idx in range(nrhs):
        cluster_id = rhs_idx // rhs_per_cluster
        if cluster_id >= n_clusters:
            cluster_id = n_clusters - 1

        # Get cluster center and add small random offset
        base_i, base_j, base_k = cluster_centers[cluster_id]
        offset = 2
        ci = base_i + np.random.randint(-offset, offset + 1)
        cj = base_j + np.random.randint(-offset, offset + 1)
        ck = base_k + np.random.randint(-offset, offset + 1)

        # Clamp to valid range
        ci = max(1, min(grid_m - 2, ci))
        cj = max(1, min(grid_m - 2, cj))
        ck = max(1, min(grid_m - 2, ck))

        cluster_info.append((cluster_id, (ci, cj, ck)))

        # Get neighbors and select subset
        neighbors = get_neighbors(ci, cj, ck)
        actual_nnz = min(
            len(neighbors), np.random.randint(4, min(9, len(neighbors) + 1))
        )
        selected = np.random.choice(len(neighbors), actual_nnz, replace=False)

        # Assign normalized weights
        if matrix_type == "complex":
            weights = np.random.rand(actual_nnz) + 1j * np.random.rand(actual_nnz) * 0.1
        else:
            weights = np.random.rand(actual_nnz)
        weights = weights / np.sum(weights)

        for w_idx, neighbor_idx in enumerate(selected):
            ni, nj, nk = neighbors[neighbor_idx]
            linear_idx = ijk_to_idx(ni, nj, nk)
            B[linear_idx, rhs_idx] = weights[w_idx]

    return B, cluster_info


def solve_superlu(A, B, n_runs=3):
    """Solve using SuperLU (factorize once, solve all RHS)."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        lu = splu(A.tocsc())
        nrhs = B.shape[1] if B.ndim > 1 else 1
        if nrhs == 1:
            x = lu.solve(B.ravel() if B.ndim > 1 else B)
        else:
            x = np.column_stack([lu.solve(B[:, i]) for i in range(nrhs)])
        times.append(time.perf_counter() - t0)
    return np.median(times), x


def solve_native_blqmr_block(A, B_all, block_size, tol=1e-8, maxiter=None, M1=None):
    """Solve all RHS using native BLQMR with given block size."""
    n = A.shape[0]
    total_rhs = B_all.shape[1]
    if maxiter is None:
        maxiter = min(n, 1000)

    n_blocks = (total_rhs + block_size - 1) // block_size

    X_all = np.zeros_like(B_all)
    total_iters = 0
    all_converged = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        for blk in range(n_blocks):
            start_col = blk * block_size
            end_col = min(start_col + block_size, total_rhs)
            B_block = B_all[:, start_col:end_col]

            x, flag, relres, niter, _ = _blqmr_python_impl(
                A, B_block, tol=tol, maxiter=maxiter, M1=M1
            )
            X_all[:, start_col:end_col] = x.reshape(-1, end_col - start_col)
            total_iters += niter
            if flag != 0:
                all_converged = False
        elapsed = time.perf_counter() - t0

    avg_iter = total_iters / n_blocks
    return elapsed, X_all, 0 if all_converged else 1, avg_iter, n_blocks


def solve_fortran_blqmr_block(
    A, B_all, block_size, tol=1e-8, maxiter=None, use_precond=False
):
    """Solve all RHS using Fortran BLQMR with given block size."""
    if not BLQMR_EXT:
        return None, None, -1, 0, 0

    n = A.shape[0]
    total_rhs = B_all.shape[1]
    if maxiter is None:
        maxiter = min(n, 1000)

    n_blocks = (total_rhs + block_size - 1) // block_size

    X_all = np.zeros_like(B_all)
    total_iters = 0
    all_converged = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        try:
            for blk in range(n_blocks):
                start_col = blk * block_size
                end_col = min(start_col + block_size, total_rhs)
                B_block = B_all[:, start_col:end_col]

                result = _blqmr_fortran(
                    A,
                    B_block,
                    tol=tol,
                    maxiter=maxiter,
                    x0=None,
                    droptol=0.001,
                    use_precond=use_precond,
                )
                X_all[:, start_col:end_col] = result.x.reshape(-1, end_col - start_col)
                total_iters += result.iter
                if result.flag != 0:
                    all_converged = False
            elapsed = time.perf_counter() - t0
        except Exception as e:
            # Print error for debugging
            print(f"[Fortran error: {e}]", end=" ")
            return None, None, -1, 0, 0

    avg_iter = total_iters / n_blocks
    return elapsed, X_all, 0 if all_converged else 1, avg_iter, n_blocks


def run_benchmark(
    A, B_all, tol=1e-8, maxiter=None, n_runs=3, matrix_type="real", M1=None
):
    """Run benchmark comparing total solve times."""
    total_rhs = B_all.shape[1]
    block_sizes = [1, 2, 4, 8, 16, 64]

    results = []

    for bs in block_sizes:
        if bs > total_rhs:
            continue

        print(f"  Block size {bs:>2}...", end=" ", flush=True)

        # Native BLQMR (no preconditioner for fair comparison)
        native_times = []
        native_info = None
        for _ in range(n_runs):
            elapsed, x, flag, avg_iter, n_blocks = solve_native_blqmr_block(
                A, B_all, bs, tol, maxiter, M1=None  # No preconditioner
            )
            native_times.append(elapsed)
            native_info = (flag, avg_iter, n_blocks)

        native_median = np.median(native_times)

        # Fortran BLQMR (no preconditioner)
        fortran_median = None
        fortran_info = (-1, 0, 0)
        if BLQMR_EXT:
            fortran_times = []
            for _ in range(n_runs):
                elapsed, x, flag, avg_iter, n_blocks = solve_fortran_blqmr_block(
                    A, B_all, bs, tol, maxiter, use_precond=False
                )
                if elapsed is not None:
                    fortran_times.append(elapsed)
                    fortran_info = (flag, avg_iter, n_blocks)
            if fortran_times:
                fortran_median = np.median(fortran_times)

        results.append(
            {
                "block_size": bs,
                "n_blocks": native_info[2],
                "native_total_ms": native_median * 1000,
                "native_avg_iter": native_info[1],
                "native_flag": native_info[0],
                "fortran_total_ms": fortran_median * 1000 if fortran_median else None,
                "fortran_avg_iter": fortran_info[1],
                "fortran_flag": fortran_info[0],
            }
        )

        status = "OK" if native_info[0] == 0 else "X"
        msg = f"Native: {native_median*1000:.1f}ms ({native_info[1]:.0f} iter/blk) {status}"
        if fortran_median:
            f_status = "OK" if fortran_info[0] == 0 else "X"
            msg += f", Fortran: {fortran_median*1000:.1f}ms ({fortran_info[1]:.0f} iter/blk) {f_status}"
        print(msg)

    return results


def print_results_table(results, matrix_type, n, total_rhs, superlu_time_ms):
    """Print benchmark results comparing total solve times."""
    print(f"\n{'='*110}")
    print(
        f"TOTAL TIME TO SOLVE {total_rhs} RHS - {matrix_type.upper()} MATRIX ({n:,} x {n:,})"
    )
    print(f"{'='*110}")

    has_fortran = any(r["fortran_total_ms"] is not None for r in results)

    # Header
    print(
        f"\n{'Method':<25} | {'Total Time':>12} | {'Iter/Blk':>10} | {'vs SuperLU':>12} | {'vs Seq(bs=1)':>12}"
    )
    print(f"{'-'*25} | {'-'*12} | {'-'*10} | {'-'*12} | {'-'*12}")

    # SuperLU baseline
    print(
        f"{'SuperLU (LU factorize)':<25} | {superlu_time_ms:>9.1f}ms | {'N/A':>10} | {'1.00x':>12} | {'-':>12}"
    )

    # Get sequential (bs=1) time for comparison
    seq_native = next((r for r in results if r["block_size"] == 1), None)
    seq_native_ms = seq_native["native_total_ms"] if seq_native else None
    seq_fortran_ms = (
        seq_native["fortran_total_ms"]
        if seq_native and seq_native["fortran_total_ms"]
        else None
    )

    print(f"{'-'*25} | {'-'*12} | {'-'*10} | {'-'*12} | {'-'*12}")

    for r in results:
        bs = r["block_size"]
        nb = r["n_blocks"]

        # Native results
        nat_ms = r["native_total_ms"]
        nat_iter = r["native_avg_iter"]
        nat_flag = r["native_flag"]

        if nat_flag == 0:
            vs_slu = superlu_time_ms / nat_ms
            vs_slu_str = f"{vs_slu:.2f}x" if vs_slu >= 1 else f"{1/vs_slu:.1f}x slower"
            if seq_native_ms and bs > 1:
                vs_seq = seq_native_ms / nat_ms
                vs_seq_str = f"{vs_seq:.2f}x"
            else:
                vs_seq_str = "baseline" if bs == 1 else "N/A"
        else:
            vs_slu_str = "FAILED"
            vs_seq_str = "FAILED"

        label = f"Native BLQMR (bs={bs}, {nb}blk)"
        print(
            f"{label:<25} | {nat_ms:>9.1f}ms | {nat_iter:>10.0f} | {vs_slu_str:>12} | {vs_seq_str:>12}"
        )

        # Fortran results
        if has_fortran and r["fortran_total_ms"] is not None:
            for_ms = r["fortran_total_ms"]
            for_iter = r["fortran_avg_iter"]
            for_flag = r["fortran_flag"]

            if for_flag == 0:
                vs_slu = superlu_time_ms / for_ms
                vs_slu_str = (
                    f"{vs_slu:.2f}x" if vs_slu >= 1 else f"{1/vs_slu:.1f}x slower"
                )
                if seq_fortran_ms and bs > 1:
                    vs_seq = seq_fortran_ms / for_ms
                    vs_seq_str = f"{vs_seq:.2f}x"
                else:
                    vs_seq_str = "baseline" if bs == 1 else "N/A"
            else:
                vs_slu_str = "FAILED"
                vs_seq_str = "FAILED"

            label = f"Fortran BLQMR (bs={bs}, {nb}blk)"
            print(
                f"{label:<25} | {for_ms:>9.1f}ms | {for_iter:>10.0f} | {vs_slu_str:>12} | {vs_seq_str:>12}"
            )


def compute_residuals(A, X, B):
    """Compute relative residuals for verification."""
    nrhs = B.shape[1]
    residuals = []
    for i in range(nrhs):
        res = np.linalg.norm(A @ X[:, i] - B[:, i]) / np.linalg.norm(B[:, i])
        residuals.append(res)
    return np.max(residuals), np.mean(residuals)


def main():
    print("=" * 80)
    print("BLQMR BLOCK SIZE BENCHMARK - TOTAL SOLVE TIME COMPARISON")
    print("=" * 80)

    info = get_backend_info()
    print(f"\nBackend Info:")
    print(f"  Fortran extension: {BLQMR_EXT}")
    print(f"  Numba acceleration: {HAS_NUMBA}")

    # Configuration
    n_target = 8000  # 20^3 = 8000
    total_rhs = 64
    tol = 1e-8
    maxiter = 1000
    n_runs = 3

    print(f"\nTest Configuration:")
    print(f"  Target matrix size: ~{n_target:,}")
    print(f"  Total RHS to solve: {total_rhs}")
    print(f"  Tolerance: {tol}")
    print(f"  Max iterations: {maxiter}")
    print(f"  Runs per test: {n_runs} (median reported)")
    print(f"  Block sizes: [1, 2, 4, 8, 16, 64]")
    print(f"  Preconditioning: None (this SPD Laplacian converges well without it)")

    # =========================================================================
    # REAL MATRICES
    # =========================================================================
    print("\n" + "=" * 80)
    print("REAL SYMMETRIC MATRIX (3D Laplacian, 7-point stencil)")
    print("=" * 80)

    A_real, n, grid_m = create_3d_fem_matrix(n_target, "real")
    B_real = create_rhs(n, total_rhs, "real")

    print(f"\nMatrix: {n}x{n} (grid={grid_m}^3), nnz={A_real.nnz:,}")

    # Create localized RHS (like point sources in FEM)
    B_real, source_info = create_localized_rhs(
        n, grid_m, total_rhs, nnz_per_rhs=6, matrix_type="real"
    )
    print(f"RHS: {n} x {total_rhs} (localized sources, ~6 nnz each, sum=1)")

    # Show a few source locations
    print(f"  Sample source locations (grid coords):", end=" ")
    for i in range(min(4, len(source_info))):
        _, ijk = source_info[i]
        print(f"{ijk}", end=" ")
    print("...")

    # SuperLU baseline
    print("\nRunning SuperLU baseline...", end=" ", flush=True)
    slu_time, x_slu = solve_superlu(A_real, B_real, n_runs)
    slu_time_ms = slu_time * 1000
    max_res, avg_res = compute_residuals(A_real, x_slu, B_real)
    print(f"done ({slu_time_ms:.1f}ms, max_residual={max_res:.2e})")

    # Block benchmark
    print("\nRunning BLQMR with different block sizes:")
    real_results = run_benchmark(A_real, B_real, tol, maxiter, n_runs, "real")

    print_results_table(real_results, "real", n, total_rhs, slu_time_ms)

    # =========================================================================
    # COMPLEX MATRICES
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPLEX SYMMETRIC MATRIX (A = A^T, not Hermitian)")
    print("=" * 80)

    A_complex, n, grid_m = create_3d_fem_matrix(n_target, "complex")
    B_complex = create_rhs(n, total_rhs, "complex")

    print(f"\nMatrix: {n}x{n} (grid={grid_m}^3), nnz={A_complex.nnz:,}")

    # Create localized RHS
    B_complex, source_info = create_localized_rhs(
        n, grid_m, total_rhs, nnz_per_rhs=6, matrix_type="complex"
    )
    print(f"RHS: {n} x {total_rhs} (localized sources, ~6 nnz each, sum=1)")

    # SuperLU baseline
    print("\nRunning SuperLU baseline...", end=" ", flush=True)
    slu_time, x_slu = solve_superlu(A_complex, B_complex, n_runs)
    slu_time_ms = slu_time * 1000
    max_res, avg_res = compute_residuals(A_complex, x_slu, B_complex)
    print(f"done ({slu_time_ms:.1f}ms, max_residual={max_res:.2e})")

    # Block benchmark
    print("\nRunning BLQMR with different block sizes:")
    complex_results = run_benchmark(
        A_complex, B_complex, tol, maxiter, n_runs, "complex"
    )

    print_results_table(complex_results, "complex", n, total_rhs, slu_time_ms)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nKey observations:")
    print("  - 'vs SuperLU': >1x means BLQMR is faster")
    print("  - 'vs Seq(bs=1)': speedup from block method vs sequential single-RHS")
    print("  - Clustered sources: RHS vectors that are spatially close may share")
    print("    solution structure, potentially benefiting block Krylov methods")
    print("  - Random sources: each RHS is independent, less shared structure")
    print("\nBlock method tradeoffs:")
    print("  - Fewer iterations with larger blocks (better Krylov subspace)")
    print("  - But O(m^2) to O(m^3) cost per iteration for block size m")
    print("  - Sweet spot depends on problem structure and implementation")


if __name__ == "__main__":
    main()
