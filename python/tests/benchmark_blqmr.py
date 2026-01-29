"""
Benchmark script for BLQMR solver comparing:
1. Native Python BLQMR
2. Fortran BLQMR (if available)
3. SciPy SuperLU direct solver

Tests both real and complex symmetric matrices across various sizes.
Uses 3D FEM-like matrices (7-point stencil Laplacian).
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve
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

    This represents a typical finite element/difference discretization:
    -u_{i-1,j,k} - u_{i+1,j,k} - u_{i,j-1,k} - u_{i,j+1,k} - u_{i,j,k-1} - u_{i,j,k+1} + 6*u_{i,j,k}

    For complex symmetric: A = A^T (not Hermitian A = A^H)
    """
    # Find grid size: n = m^3
    m = max(2, int(round(n_target ** (1 / 3))))
    n = m * m * m

    dtype = np.complex128 if matrix_type == "complex" else np.float64

    # Diagonal values - for complex symmetric, use complex values (not conjugate pairs)
    diag_main = 6.0 + 0.2j if matrix_type == "complex" else 6.0
    diag_off = -1.0 + 0.1j if matrix_type == "complex" else -1.0

    # Build sparse matrix using COO format for efficiency
    row, col, data = [], [], []

    for i in range(m):
        for j in range(m):
            for k in range(m):
                idx = i * m * m + j * m + k

                # Main diagonal
                row.append(idx)
                col.append(idx)
                data.append(diag_main)

                # x-direction neighbors (i±1)
                if i > 0:
                    row.append(idx)
                    col.append(idx - m * m)
                    data.append(diag_off)
                if i < m - 1:
                    row.append(idx)
                    col.append(idx + m * m)
                    data.append(diag_off)

                # y-direction neighbors (j±1)
                if j > 0:
                    row.append(idx)
                    col.append(idx - m)
                    data.append(diag_off)
                if j < m - 1:
                    row.append(idx)
                    col.append(idx + m)
                    data.append(diag_off)

                # z-direction neighbors (k±1)
                if k > 0:
                    row.append(idx)
                    col.append(idx - 1)
                    data.append(diag_off)
                if k < m - 1:
                    row.append(idx)
                    col.append(idx + 1)
                    data.append(diag_off)

    A = sparse.coo_matrix((data, (row, col)), shape=(n, n), dtype=dtype).tocsc()

    # Verify symmetry (A = A^T)
    sym_err = sparse.linalg.norm(A - A.T)
    assert sym_err < 1e-14, f"Matrix not symmetric! Error: {sym_err}"

    return A, n, m


def benchmark_superlu(A, B, n_runs=3):
    """Benchmark SciPy SuperLU direct solver for multiple RHS."""
    nrhs = B.shape[1] if B.ndim > 1 else 1
    times = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        if nrhs == 1:
            x = spsolve(A, B.ravel() if B.ndim > 1 else B)
        else:
            # SuperLU factorize once, solve multiple RHS
            lu = splu(A.tocsc())
            x = np.column_stack([lu.solve(B[:, i]) for i in range(nrhs)])
        times.append(time.perf_counter() - t0)

    return np.median(times), x


def benchmark_native_blqmr(A, B, tol=1e-8, maxiter=None, n_runs=3):
    """Benchmark native Python BLQMR with multiple RHS (block method)."""
    n = A.shape[0]
    nrhs = B.shape[1] if B.ndim > 1 else 1

    if maxiter is None:
        maxiter = min(n, 1000)

    # NOTE: We run WITHOUT preconditioner because the native BLQMR converges
    # well without it for symmetric positive definite matrices.
    # The ILU(0) from scipy.spilu is too aggressive and hurts convergence.
    M1 = None

    # Ensure B is 2D for block method
    B_block = B.reshape(-1, nrhs) if B.ndim == 1 else B

    # Warmup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _blqmr_python_impl(A, B_block, tol=tol, maxiter=min(10, maxiter), M1=M1)
        except:
            pass

    times, results = [], []
    for _ in range(n_runs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            try:
                x, flag, relres, niter, _ = _blqmr_python_impl(
                    A, B_block, tol=tol, maxiter=maxiter, M1=M1
                )
                elapsed = time.perf_counter() - t0
                if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                    flag = 4
                times.append(elapsed)
                results.append((x, flag, niter, relres))
            except Exception as e:
                times.append(time.perf_counter() - t0)
                results.append((np.zeros_like(B_block), 5, 0, 1.0))

    if not times:
        return None, (np.zeros_like(B_block), -1, 0, 1.0)

    idx = np.argmin(np.abs(np.array(times) - np.median(times)))
    return np.median(times), results[idx]


def benchmark_fortran_blqmr(A, B, tol=1e-8, maxiter=None, n_runs=3):
    """Benchmark Fortran BLQMR with multiple RHS."""
    if not BLQMR_EXT:
        return None, None

    n = A.shape[0]
    nrhs = B.shape[1] if B.ndim > 1 else 1

    if maxiter is None:
        maxiter = min(n, 1000)

    # Warmup
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _blqmr_fortran(
                A,
                B,
                tol=tol,
                maxiter=min(10, maxiter),
                x0=None,
                droptol=0.001,
                precond_type=False,
            )
    except:
        pass

    times, results = [], []
    for _ in range(n_runs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            try:
                result = _blqmr_fortran(
                    A,
                    B,
                    tol=tol,
                    maxiter=maxiter,
                    x0=None,
                    droptol=0.001,
                    precond_type=False,
                )
                elapsed = time.perf_counter() - t0
                if np.any(np.isnan(result.x)) or np.any(np.isinf(result.x)):
                    times.append(elapsed)
                    results.append((result.x, 4, result.iter, result.relres))
                else:
                    times.append(elapsed)
                    results.append((result.x, result.flag, result.iter, result.relres))
            except Exception as e:
                return None, None

    if not times:
        return None, None

    idx = np.argmin(np.abs(np.array(times) - np.median(times)))
    return np.median(times), results[idx]


def run_benchmark(sizes, matrix_type="real", nrhs=4, tol=1e-8, n_runs=3):
    """Run benchmarks for all solvers across matrix sizes."""
    results = []

    for n_target in sizes:
        print(f"  Testing n≈{n_target:,} (nrhs={nrhs})...", end=" ", flush=True)

        A, n_actual, grid_m = create_3d_fem_matrix(n_target, matrix_type)

        # Multiple RHS - random but reproducible
        np.random.seed(42)
        if matrix_type == "complex":
            B = np.random.randn(n_actual, nrhs) + 1j * np.random.randn(n_actual, nrhs)
        else:
            B = np.random.randn(n_actual, nrhs)

        # SuperLU (direct solver)
        try:
            t_superlu, x_superlu = benchmark_superlu(A, B, n_runs)
            res_superlu = max(
                np.linalg.norm(A @ x_superlu[:, i] - B[:, i]) / np.linalg.norm(B[:, i])
                for i in range(nrhs)
            )
        except Exception as e:
            t_superlu, res_superlu = None, None

        # Native BLQMR
        try:
            t_native, (
                x_native,
                flag_native,
                iter_native,
                relres_native,
            ) = benchmark_native_blqmr(A, B, tol, n_runs=n_runs)
            if (
                flag_native in (4, 5)
                or np.any(np.isnan(x_native))
                or np.any(np.isinf(x_native))
            ):
                res_native = float("inf")
            else:
                x_nat = x_native.reshape(n_actual, -1)
                res_native = max(
                    np.linalg.norm(A @ x_nat[:, i] - B[:, i]) / np.linalg.norm(B[:, i])
                    for i in range(min(nrhs, x_nat.shape[1]))
                )
        except Exception as e:
            t_native, flag_native, iter_native, res_native = None, -1, 0, None

        # Fortran BLQMR (now supports both real and complex)
        t_fortran, flag_fortran, iter_fortran, res_fortran = None, -1, 0, None
        if BLQMR_EXT:
            try:
                result = benchmark_fortran_blqmr(A, B, tol, n_runs=n_runs)
                if result[0] is not None:
                    t_fortran, (x_fortran, flag_fortran, iter_fortran, _) = result
                    if flag_fortran in (4, 5) or np.any(np.isnan(x_fortran)):
                        res_fortran = float("inf")
                    else:
                        x_for = x_fortran.reshape(n_actual, -1)
                        res_fortran = max(
                            np.linalg.norm(A @ x_for[:, i] - B[:, i])
                            / np.linalg.norm(B[:, i])
                            for i in range(min(nrhs, x_for.shape[1]))
                        )
            except:
                pass

        results.append(
            {
                "n": n_actual,
                "grid": grid_m,
                "nnz": A.nnz,
                "nrhs": nrhs,
                "t_superlu": t_superlu,
                "res_superlu": res_superlu,
                "t_native": t_native,
                "flag_native": flag_native,
                "iter_native": iter_native,
                "res_native": res_native,
                "t_fortran": t_fortran,
                "flag_fortran": flag_fortran,
                "iter_fortran": iter_fortran,
                "res_fortran": res_fortran,
            }
        )
        print(f"done (grid={grid_m}³={n_actual}, nnz={A.nnz:,})")

    return results


def print_results_table(results, matrix_type):
    """Print benchmark results in a formatted table."""
    print(f"\n{'='*130}")
    print(
        f"BENCHMARK RESULTS - {matrix_type.upper()} MATRICES (3D FEM 7-point stencil)"
    )
    print(f"{'='*130}")
    print(
        "Flags: 0=converged, 1=maxiter, 2=precond fail, 3=stagnated, 4=NaN/Inf, 5=error"
    )
    print()

    # Header
    print(
        f"{'Grid':>6} {'Size':>8} {'NNZ':>10} │ {'SuperLU':>10} │ {'Native BLQMR':>12} {'Iter':>5} {'Flg':>3} │ "
        f"{'Fortran BLQMR':>13} {'Iter':>5} {'Flg':>3} │ {'Native/SLU':>12} {'Fort/SLU':>12}"
    )
    print(
        f"{'-'*6} {'-'*8} {'-'*10} │ {'-'*10} │ {'-'*12} {'-'*5} {'-'*3} │ {'-'*13} {'-'*5} {'-'*3} │ {'-'*12} {'-'*12}"
    )

    for r in results:
        grid = f"{r['grid']}³"
        n = r["n"]
        nnz = r["nnz"]

        t_slu = f"{r['t_superlu']*1000:.1f}ms" if r["t_superlu"] else "N/A"
        t_nat = f"{r['t_native']*1000:.1f}ms" if r["t_native"] else "N/A"
        t_for = f"{r['t_fortran']*1000:.1f}ms" if r["t_fortran"] else "N/A"

        iter_nat = str(r["iter_native"]) if r["t_native"] else "-"
        flag_nat = str(r["flag_native"]) if r["t_native"] else "-"
        iter_for = str(r["iter_fortran"]) if r["t_fortran"] else "-"
        flag_for = str(r["flag_fortran"]) if r["t_fortran"] else "-"

        # Speedup ratios
        if r["t_superlu"] and r["t_native"] and r["t_native"] > 0:
            ratio_nat = r["t_superlu"] / r["t_native"]
            speedup_nat = (
                f"{ratio_nat:.2f}x" if ratio_nat >= 1 else f"{1/ratio_nat:.1f}x slower"
            )
        else:
            speedup_nat = "N/A"

        if r["t_superlu"] and r["t_fortran"] and r["t_fortran"] > 0:
            ratio_for = r["t_superlu"] / r["t_fortran"]
            speedup_for = (
                f"{ratio_for:.2f}x" if ratio_for >= 1 else f"{1/ratio_for:.1f}x slower"
            )
        else:
            speedup_for = "N/A"

        print(
            f"{grid:>6} {n:>8,} {nnz:>10,} │ {t_slu:>10} │ {t_nat:>12} {iter_nat:>5} {flag_nat:>3} │ "
            f"{t_for:>13} {iter_for:>5} {flag_for:>3} │ {speedup_nat:>12} {speedup_for:>12}"
        )

    # Summary
    print(f"\n{'─'*130}")
    print("SPEEDUP SUMMARY (ratio > 1 means BLQMR faster than SuperLU):")
    print("  (Only counting runs where solver converged: flag=0)")

    native_speedups = [
        r["t_superlu"] / r["t_native"]
        for r in results
        if r["t_superlu"] and r["t_native"] and r["flag_native"] == 0
    ]
    fortran_speedups = [
        r["t_superlu"] / r["t_fortran"]
        for r in results
        if r["t_superlu"] and r["t_fortran"] and r["flag_fortran"] == 0
    ]

    if native_speedups:
        print(
            f"  Native BLQMR:  avg={np.mean(native_speedups):.2f}x, "
            f"min={min(native_speedups):.2f}x, max={max(native_speedups):.2f}x"
        )
    else:
        print("  Native BLQMR:  No converged runs")

    if fortran_speedups:
        print(
            f"  Fortran BLQMR: avg={np.mean(fortran_speedups):.2f}x, "
            f"min={min(fortran_speedups):.2f}x, max={max(fortran_speedups):.2f}x"
        )
    else:
        print("  Fortran BLQMR: No converged runs (or not available)")


def print_residual_table(results, matrix_type):
    """Print residual accuracy comparison."""
    print(f"\n{'='*90}")
    print(f"RESIDUAL ACCURACY - {matrix_type.upper()} MATRICES")
    print(f"{'='*90}")
    print(
        f"{'Grid':>6} {'Size':>8} │ {'SuperLU':>12} │ {'Native BLQMR':>12} │ {'Fortran BLQMR':>13}"
    )
    print(f"{'-'*6} {'-'*8} │ {'-'*12} │ {'-'*12} │ {'-'*13}")

    for r in results:
        grid = f"{r['grid']}³"
        res_slu = f"{r['res_superlu']:.2e}" if r["res_superlu"] is not None else "N/A"
        res_nat = (
            f"{r['res_native']:.2e}"
            if r["res_native"] is not None and r["res_native"] != float("inf")
            else "FAILED"
        )
        res_for = (
            f"{r['res_fortran']:.2e}"
            if r["res_fortran"] is not None and r["res_fortran"] != float("inf")
            else "N/A"
        )
        print(f"{grid:>6} {r['n']:>8,} │ {res_slu:>12} │ {res_nat:>12} │ {res_for:>13}")


def main():
    print("=" * 90)
    print("BLQMR BENCHMARK: Native Python vs Fortran vs SciPy SuperLU")
    print("=" * 90)

    info = get_backend_info()
    print(f"\nBackend Info:")
    print(f"  Fortran extension available: {BLQMR_EXT}")
    print(f"  Numba acceleration available: {HAS_NUMBA}")
    print(f"  Active backend: {info['backend']}")

    # Matrix sizes (these become grid³ for 3D)
    # Grid sizes: 5,8,10,15,20,25,30,40 -> n = 125, 512, 1000, 3375, 8000, 15625, 27000, 64000
    sizes = [125, 512, 1000, 3375, 8000, 15625, 27000, 64000]
    nrhs = 4  # Number of right-hand sides (block size)

    print(f"\n3D FEM Test Configuration:")
    print(f"  Target sizes: {sizes}")
    print(f"  Block size (nrhs): {nrhs}")
    print(f"  Tolerance: 1e-8")
    print(f"  Max iterations: 1000")
    print(f"  Runs per test: 3 (median time reported)")

    # Real matrices
    print("\n" + "─" * 90)
    print("Testing REAL symmetric matrices (3D Laplacian, 7-point stencil)...")
    print("─" * 90)
    real_results = run_benchmark(sizes, "real", nrhs=nrhs, tol=1e-8, n_runs=3)
    print_results_table(real_results, "real")
    print_residual_table(real_results, "real")

    # Complex symmetric matrices
    print("\n" + "─" * 90)
    print("Testing COMPLEX SYMMETRIC matrices (3D Laplacian, 7-point stencil)...")
    print("  Note: A = A^T (symmetric), NOT A = A^H (Hermitian)")
    print("─" * 90)
    complex_results = run_benchmark(sizes, "complex", nrhs=nrhs, tol=1e-8, n_runs=3)
    print_results_table(complex_results, "complex")
    print_residual_table(complex_results, "complex")

    # Summary
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print(f"Fortran extension: {'AVAILABLE' if BLQMR_EXT else 'NOT AVAILABLE'}")
    print(f"Numba JIT: {'ENABLED' if HAS_NUMBA else 'DISABLED'}")
    print(f"\nTest matrix: 3D Laplacian (7-point stencil) - typical FEM structure")
    print(f"  - Symmetric: A = A^T")
    print(f"  - Sparse: ~7 nonzeros per row")
    print(f"  - Condition number: O(n^{2/3}) for 3D")
    print(f"\nNotes:")
    print(f"  - Flag 0=converged, 1=maxiter, 3=stagnated, 4=NaN/Inf")
    print(f"  - Block BLQMR processes {nrhs} RHS simultaneously")
    print(
        f"  - Iterative solvers benefit from: large sparse systems, good preconditioners"
    )
    print(f"  - Direct solvers (SuperLU) have O(n^2) complexity for 3D problems")


if __name__ == "__main__":
    main()
