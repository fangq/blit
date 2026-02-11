"""
Benchmark: Single-threaded vs OpenMP multi-threaded BLQMR (Fortran backend).

Sweeps nblock values for a fixed 64-RHS complex symmetric Helmholtz system.
Compares wall-clock time and verifies solution accuracy.

Usage:
    python benchmark_omp.py
    OMP_NUM_THREADS=4 python benchmark_omp.py
"""

import numpy as np
from scipy import sparse
import time
import sys
import os
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from blocksolver import BLQMR_EXT, BLQMR_OMP
from blocksolver.blqmr import _blqmr_fortran

# Reuse Helmholtz assembly from the RHS sweep benchmark
from benchmark_rhs_sweep import (
    create_helmholtz_system,
    create_distributed_sources,
    max_residual,
    _complex_to_real_system,
    _real_to_complex_solution,
)

# PARDISO detection
HAS_PARDISO = False
pypardiso_solver = None

try:
    from pypardiso import PyPardisoSolver

    pypardiso_solver = PyPardisoSolver()
    HAS_PARDISO = True
except Exception:
    pass


def _set_blas_threads(n):
    """Set BLAS thread count at runtime via library APIs (not just env vars)."""
    # MKL runtime API
    try:
        import ctypes

        mkl_rt = ctypes.CDLL("libmkl_rt.so", mode=ctypes.RTLD_GLOBAL)
        mkl_rt.MKL_Set_Num_Threads(ctypes.c_int(n))
    except (OSError, AttributeError):
        pass

    # OpenBLAS runtime API
    try:
        import ctypes

        openblas = ctypes.CDLL("libopenblas.so", mode=ctypes.RTLD_GLOBAL)
        openblas.openblas_set_num_threads(ctypes.c_int(n))
    except (OSError, AttributeError):
        pass

    # Try via numpy's BLAS if available (NumPy >= 1.25)
    try:
        import numpy._core._multiarray_umath as _mu

        if hasattr(_mu, "_set_num_threads"):
            _mu._set_num_threads(n)
    except (ImportError, AttributeError):
        pass


def _get_blas_threads():
    """Get current BLAS thread count via library APIs."""
    try:
        import ctypes

        mkl_rt = ctypes.CDLL("libmkl_rt.so", mode=ctypes.RTLD_GLOBAL)
        return mkl_rt.MKL_Get_Max_Threads()
    except (OSError, AttributeError):
        pass

    try:
        import ctypes

        openblas = ctypes.CDLL("libopenblas.so", mode=ctypes.RTLD_GLOBAL)
        return openblas.openblas_get_num_threads()
    except (OSError, AttributeError):
        pass

    return 1


# Try threadpoolctl if available (most reliable cross-platform approach)
_threadpoolctl = None
try:
    import threadpoolctl

    _threadpoolctl = threadpoolctl
except ImportError:
    pass


def benchmark_blqmr_fortran(A, B, nblock=0, tol=1e-6, maxiter=2000, n_runs=3):
    """
    Benchmark Fortran BLQMR with a given nblock value.

    Parameters
    ----------
    nblock : int
        0 = single block (no OMP splitting)
        >0 = partition RHS into chunks of this size, solve in parallel
    """
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _blqmr_fortran(
                A,
                B,
                tol=tol,
                maxiter=maxiter,
                x0=None,
                droptol=0.001,
                precond_type=3,  # Jacobi-split
                nblock=nblock,
            )
        times.append(time.perf_counter() - t0)

    return np.min(times), result


def benchmark_direct(A, B, n_runs=3):
    """
    Benchmark direct solvers: PARDISO (if available) and SuperLU.

    For complex matrices with PARDISO: converts to real 2x2 block system.
    Returns dict of {solver_name: (time, x, max_relres)}.
    """
    from scipy.sparse.linalg import splu

    if B.ndim == 1:
        B = B.reshape(-1, 1)
    n = A.shape[0]
    nrhs = B.shape[1]
    results = {}

    # SuperLU: factorize once, solve all RHS
    superlu_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        lu = splu(A.tocsc())
        x = np.zeros((n, nrhs), dtype=B.dtype)
        for i in range(nrhs):
            x[:, i] = lu.solve(B[:, i])
        superlu_times.append(time.perf_counter() - t0)
    t_superlu = np.min(superlu_times)
    res_superlu = max_residual(A, x, B)
    results["SuperLU"] = (t_superlu, x, res_superlu)

    # PARDISO (if available)
    if HAS_PARDISO:
        is_complex = np.iscomplexobj(A) or np.iscomplexobj(B)
        pardiso_times = []
        pardiso_ok = False

        # PARDISO can fail with error -3 when nested OpenMP/BLAS threading
        # conflicts during reordering. Try current settings first, then
        # force single-threaded BLAS via runtime API.
        for attempt in range(2):
            if attempt == 1:
                # Force single-threaded BLAS via runtime APIs (not just env vars)
                if _threadpoolctl is not None:
                    _tpc_ctx = _threadpoolctl.threadpool_limits(limits=1)
                else:
                    _set_blas_threads(1)

            try:
                # Need fresh solver instance after thread change
                from pypardiso import PyPardisoSolver

                ps = PyPardisoSolver()

                if is_complex:
                    A_r = A.real
                    A_i = A.imag
                    A_real = sparse.bmat([[A_r, -A_i], [A_i, A_r]], format="csr")
                    B_real = np.vstack([B.real, B.imag])

                    # Test factorization
                    ps.solve(A_real, B_real[:, 0:1])

                    pardiso_times = []
                    for _ in range(n_runs):
                        t0 = time.perf_counter()
                        x_real = ps.solve(A_real, B_real)
                        pardiso_times.append(time.perf_counter() - t0)

                    if x_real.ndim == 1:
                        x_real = x_real.reshape(-1, 1)
                    x = x_real[:n, :] + 1j * x_real[n:, :]
                else:
                    A_csr = A.tocsr()
                    ps.solve(A_csr, B[:, 0:1])  # test

                    pardiso_times = []
                    for _ in range(n_runs):
                        t0 = time.perf_counter()
                        x = ps.solve(A_csr, B)
                        pardiso_times.append(time.perf_counter() - t0)

                pardiso_ok = True
                solver_name = "PARDISO"

            except Exception as e:
                if attempt == 0:
                    pardiso_times = []
                else:
                    print(f"    PARDISO failed: {e}")
            finally:
                if attempt == 1:
                    # Restore BLAS threads
                    if _threadpoolctl is not None:
                        try:
                            _tpc_ctx.restore_original_limits()
                        except Exception:
                            pass
                    else:
                        omp_t = os.environ.get("OMP_NUM_THREADS", None)
                        _set_blas_threads(int(omp_t) if omp_t else os.cpu_count())

        if pardiso_ok:
            t_pardiso = np.min(pardiso_times)
            res_pardiso = max_residual(A, x, B)
            results[solver_name] = (t_pardiso, x, res_pardiso)

    return results


def run_benchmark(grid_size=20, nrhs=64, tol=1e-6, maxiter=2000, n_runs=3):
    """Run single vs multi-threaded comparison."""

    # Try to read OMP_NUM_THREADS
    omp_threads_env = os.environ.get("OMP_NUM_THREADS", None)
    # Fallback: os.cpu_count() is what OpenMP typically defaults to
    actual_threads = int(omp_threads_env) if omp_threads_env else os.cpu_count()

    print("=" * 80)
    print("BENCHMARK: Single-threaded vs OpenMP BLQMR (Fortran)")
    print("=" * 80)
    print(f"\n  Fortran backend:  {BLQMR_EXT}")
    print(f"  OpenMP support:   {BLQMR_OMP}")
    print(f"  PARDISO:          {HAS_PARDISO}")
    print(f"  OMP_NUM_THREADS:  {omp_threads_env or 'not set (default: all cores)'}")
    print(f"  Effective threads: {actual_threads}")
    print(f"  Timing runs:      {n_runs} (report minimum)")

    if not BLQMR_EXT:
        print("\nERROR: Fortran backend not available. Build with f2py first.")
        return
    if not BLQMR_OMP:
        print("\nERROR: OpenMP wrappers not found in Fortran module.")
        print("       Rebuild with: USE_OPENMP=1 make")
        return

    # Create system
    print(f"\n  Creating Helmholtz FEM system (grid {grid_size}³)...")
    A, node, elem, M1, M2, n = create_helmholtz_system(grid_size)
    print(f"  Matrix: n={n}, nnz={A.nnz}, complex symmetric")

    print(f"  Creating {nrhs} distributed sources...")
    B = create_distributed_sources(node, elem, nrhs)
    print(f"  RHS shape: {B.shape}")

    # nblock sweep: 0 (no split), 1, 2, 4, 8, 16, 32, 64
    nblock_values = [0, 1, 2, 4, 8, 16, 32, nrhs]
    # Remove duplicates and values > nrhs
    nblock_values = sorted(set(nb for nb in nblock_values if nb <= nrhs))

    print(f"\n  Solver config: tol={tol}, maxiter={maxiter}, precond=Jacobi-split")
    print(f"  nblock values: {nblock_values}")
    print(f"    nblock=0  -> all {nrhs} RHS in one block (no OMP)")
    print(f"    nblock=k  -> chunks of k, solved in parallel")
    print(f"    nblock={nrhs} -> same as nblock=0 (nblock >= nrhs)")

    # ── Direct solvers ──
    print(f"\n  Running direct solvers...")
    direct_results = benchmark_direct(A, B, n_runs=n_runs)
    for name, (t, x, res) in direct_results.items():
        print(f"    {name}: {t:.4f}s (max relres={res:.2e})")

    # ── Serial baseline: solve each RHS one at a time ──
    print(f"\n  Running serial baseline (1 RHS per call, {nrhs} calls)...")
    serial_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        x_serial = np.zeros((n, nrhs), dtype=B.dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(nrhs):
                result_i = _blqmr_fortran(
                    A,
                    B[:, i : i + 1],
                    tol=tol,
                    maxiter=maxiter,
                    x0=None,
                    droptol=0.001,
                    precond_type=3,
                    nblock=0,
                )
                xi = result_i.x
                x_serial[:, i] = xi.ravel()
        serial_times.append(time.perf_counter() - t0)

    baseline_time = np.min(serial_times)
    baseline_x = x_serial.copy()
    baseline_res = max_residual(A, baseline_x, B)

    # ── Sweep nblock values ──
    print(f"\n{'─' * 90}")
    print(
        f"{'Method':>12} │ {'Time (s)':>10} │ {'Speedup':>8} │ {'Iters':>6} │ {'Flag':>5} │ {'Max Relres':>12}"
    )
    print(f"{'─' * 90}")

    # Print serial baseline row
    print(
        f"{'serial 1x' + str(nrhs):>12} │ {baseline_time:>10.4f} │ {'1.00x':>8} │ {'—':>6} │ {'—':>5} │ {baseline_res:>12.2e}"
    )

    # Print direct solver rows
    for name, (t, x, res) in direct_results.items():
        speedup = baseline_time / t if t > 0 else 0.0
        print(
            f"{name:>12} │ {t:>10.4f} │ {speedup:>7.2f}x │ {'—':>6} │ {'—':>5} │ {res:>12.2e}"
        )

    results = []

    for nb in nblock_values:
        label = f"nblock={nb}" if nb > 0 else "nblock=0"
        t, result = benchmark_blqmr_fortran(
            A, B, nblock=nb, tol=tol, maxiter=maxiter, n_runs=n_runs
        )

        x = result.x
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        res = max_residual(A, x, B) if result.flag <= 1 else float("nan")
        speedup = baseline_time / t if t > 0 else 0.0

        flag_str = "OK" if result.flag == 0 else f"f={result.flag}"
        print(
            f"{label:>12} │ {t:>10.4f} │ {speedup:>7.2f}x │ {result.iter:>6} │ {flag_str:>5} │ {res:>12.2e}"
        )

        results.append(
            {
                "nblock": nb,
                "time": t,
                "speedup": speedup,
                "iter": result.iter,
                "flag": result.flag,
                "relres": result.relres,
                "max_res": res,
            }
        )

    print(f"{'─' * 90}")

    # Verify solutions match serial baseline
    print(f"\nSolution consistency check (vs serial 1-RHS baseline):")
    for r in results:
        if r["flag"] > 1:
            continue
        t, result = benchmark_blqmr_fortran(
            A, B, nblock=r["nblock"], tol=tol, maxiter=maxiter, n_runs=1
        )
        x = result.x
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        diff = np.max(np.abs(x - baseline_x)) / (np.max(np.abs(baseline_x)) + 1e-300)
        status = "PASS" if diff < 100 * tol else "WARN"
        print(f"  nblock={r['nblock']:>3}: max relative diff = {diff:.2e}  [{status}]")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Serial baseline (1 RHS x {nrhs} calls): {baseline_time:.4f}s")

    for name, (t, x, res) in direct_results.items():
        speedup = baseline_time / t if t > 0 else 0.0
        print(f"  {name}: {t:.4f}s  ({speedup:.2f}x vs serial)")

    valid = [r for r in results if r["flag"] <= 1]
    if valid:
        best = min(valid, key=lambda r: r["time"])
        worst = max(valid, key=lambda r: r["time"])
        print(
            f"  Fastest BLQMR:  nblock={best['nblock']:>3}  {best['time']:.4f}s  ({best['speedup']:.2f}x vs serial)"
        )
        print(
            f"  Slowest BLQMR:  nblock={worst['nblock']:>3}  {worst['time']:.4f}s  ({worst['speedup']:.2f}x vs serial)"
        )

        # Block QMR benefit (nblock=0 vs serial)
        nb0 = next((r for r in valid if r["nblock"] == 0), None)
        if nb0:
            print(f"\n  Block QMR speedup (nblock=0 vs serial):  {nb0['speedup']:.2f}x")

        # Best OMP benefit
        omp_results = [r for r in valid if 0 < r["nblock"] < nrhs]
        if omp_results:
            best_omp = min(omp_results, key=lambda r: r["time"])
            print(
                f"  Best OMP speedup (nblock={best_omp['nblock']} vs serial): {best_omp['speedup']:.2f}x"
            )
            if nb0:
                omp_vs_block = nb0["time"] / best_omp["time"]
                print(
                    f"  OMP vs single-block (nblock={best_omp['nblock']} vs nblock=0): {omp_vs_block:.2f}x"
                )

        # Compare best BLQMR vs direct solvers
        best_t = best["time"]
        for name, (t, x, res) in direct_results.items():
            ratio = t / best_t if best_t > 0 else 0.0
            winner = "BLQMR" if ratio > 1.0 else name
            print(f"\n  Best BLQMR vs {name}: {ratio:.2f}x ({winner} wins)")

    else:
        print("  No valid BLQMR results!")

    return results


if __name__ == "__main__":
    # Default: 20³ grid, 64 RHS
    # Override with: python benchmark_omp.py <grid_size> <nrhs>
    grid_size = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    nrhs = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    run_benchmark(grid_size=grid_size, nrhs=nrhs)
