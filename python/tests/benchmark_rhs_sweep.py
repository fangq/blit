"""
Benchmark sweep: Grid size vs RHS count
Find crossover points where BLQMR beats direct solver.

Uses block size of 4 for BLQMR.
Complex symmetric Helmholtz FEM matrix with split Jacobi preconditioner.
Matches block_size benchmark configuration.

Direct solver: PARDISO (if available), otherwise SuperLU
BLQMR: Fortran (if available), otherwise Native Python
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
import time
import sys
import os
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from blocksolver import BLQMR_EXT, HAS_NUMBA
from blocksolver.blqmr import _blqmr_python_impl

if BLQMR_EXT:
    from blocksolver.blqmr import _blqmr_fortran

# PARDISO will be checked lazily in benchmark_direct()
HAS_PARDISO = False
pypardiso_solver = None


def _set_blas_threads(n):
    """Set BLAS thread count at runtime via library APIs."""
    try:
        import ctypes

        mkl_rt = ctypes.CDLL("libmkl_rt.so", mode=ctypes.RTLD_GLOBAL)
        mkl_rt.MKL_Set_Num_Threads(ctypes.c_int(n))
        return
    except (OSError, AttributeError):
        pass
    try:
        import ctypes

        openblas = ctypes.CDLL("libopenblas.so", mode=ctypes.RTLD_GLOBAL)
        openblas.openblas_set_num_threads(ctypes.c_int(n))
        return
    except (OSError, AttributeError):
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


_threadpoolctl = None
try:
    import threadpoolctl

    _threadpoolctl = threadpoolctl
except ImportError:
    pass


def _pardiso_solve_safe(A_csr, B):
    """
    Solve with PARDISO, retrying with single-threaded BLAS if error -3 occurs.

    PARDISO error -3 (reordering failure) is typically caused by nested
    OpenMP thread contention when BLAS also uses multiple threads.

    Reuses a single solver instance to avoid memory leaks from repeated
    PyPardisoSolver allocations.
    """
    global _pardiso_instance
    from pypardiso import PyPardisoSolver

    saved_threads = _get_blas_threads()

    for attempt in range(2):
        try:
            # Reuse global instance — free previous factorization first
            if _pardiso_instance is not None:
                try:
                    _pardiso_instance.free_memory(everything=True)
                except Exception:
                    pass
            _pardiso_instance = PyPardisoSolver()
            x = _pardiso_instance.solve(A_csr, B)
            return x
        except Exception:
            if attempt == 0:
                # Pin BLAS to 1 thread and retry
                if _threadpoolctl is not None:
                    _threadpoolctl.threadpool_limits(limits=1)
                else:
                    _set_blas_threads(1)
            else:
                # Restore and re-raise
                if _threadpoolctl is None:
                    _set_blas_threads(saved_threads)
                raise


# Global PARDISO instance to avoid memory leaks
_pardiso_instance = None


def _try_init_pardiso():
    """Try to initialize PARDISO solver (lazy initialization)."""
    global HAS_PARDISO, pypardiso_solver
    if pypardiso_solver is not None:
        return HAS_PARDISO
    try:
        from pypardiso import PyPardisoSolver

        pypardiso_solver = PyPardisoSolver()
        HAS_PARDISO = True
    except Exception:
        HAS_PARDISO = False
    return HAS_PARDISO


def meshgrid6(x, y, z):
    """
    Generate tetrahedral mesh from regular grid (6 tets per cube).
    Matches MATLAB's meshgrid6 function.
    """
    nx, ny, nz = len(x), len(y), len(z)

    # Create nodes
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    node = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Create elements (6 tets per cube)
    elem_list = []

    def idx(i, j, k):
        return i * ny * nz + j * nz + k

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                # 8 corners of the cube
                n0 = idx(i, j, k)
                n1 = idx(i + 1, j, k)
                n2 = idx(i + 1, j + 1, k)
                n3 = idx(i, j + 1, k)
                n4 = idx(i, j, k + 1)
                n5 = idx(i + 1, j, k + 1)
                n6 = idx(i + 1, j + 1, k + 1)
                n7 = idx(i, j + 1, k + 1)

                # 6 tetrahedra (same decomposition as MATLAB meshgrid6)
                elem_list.append([n0, n1, n3, n4])
                elem_list.append([n1, n2, n3, n6])
                elem_list.append([n1, n4, n5, n6])
                elem_list.append([n3, n4, n6, n7])
                elem_list.append([n1, n3, n4, n6])
                elem_list.append([n1, n2, n6, n5])

    elem = np.array(elem_list, dtype=np.int32)
    return node, elem


def assemble_helmholtz_fem(node, elem):
    """
    Assemble complex symmetric Helmholtz-like FEM matrix.
    Matches MATLAB's assemble_helmholtz_fem function.

    Creates: A = K - 1.0*M + 0.3i*M + regularization
    where K is stiffness, M is mass matrix.
    """
    n = node.shape[0]
    nelem = elem.shape[0]

    # Preallocate COO arrays
    max_entries = 16 * nelem
    II = np.zeros(max_entries, dtype=np.int32)
    JJ = np.zeros(max_entries, dtype=np.int32)
    VV = np.zeros(max_entries, dtype=np.complex128)
    cnt = 0

    for e in range(nelem):
        idx = elem[e, :]
        coords = node[idx, :]

        # Jacobian
        d1 = coords[1, :] - coords[0, :]
        d2 = coords[2, :] - coords[0, :]
        d3 = coords[3, :] - coords[0, :]
        J = np.column_stack([d1, d2, d3])

        vol = abs(np.linalg.det(J)) / 6.0
        if vol < 1e-15:
            continue

        # Gradient of shape functions
        invJ = np.linalg.inv(J)
        grad_ref = np.array(
            [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        ).T
        grad_N = invJ.T @ grad_ref

        # Element stiffness and mass
        Ke = vol * (grad_N.T @ grad_N)
        Me = vol / 20.0 * (np.ones((4, 4)) + np.eye(4))

        # Helmholtz with absorption (complex symmetric, NOT Hermitian)
        # A = K - k^2 * M + i * sigma * M
        Ae = Ke - 1.0 * Me + 0.3j * Me

        # Assemble
        for i in range(4):
            for j in range(4):
                II[cnt] = idx[i]
                JJ[cnt] = idx[j]
                VV[cnt] = Ae[i, j]
                cnt += 1

    II = II[:cnt]
    JJ = JJ[:cnt]
    VV = VV[:cnt]

    A = sparse.coo_matrix((VV, (II, JJ)), shape=(n, n)).tocsr()

    # Symmetrize and add regularization
    A = (A + A.T) / 2
    A = A + 0.01 * np.mean(np.abs(A.diagonal())) * sparse.eye(n, format="csr")

    return A.tocsc()


def create_distributed_sources(node, elem, nrhs):
    """
    Create spatially distributed point sources using barycentric coordinates.
    Matches MATLAB's create_distributed_sources function.
    """
    from scipy.spatial import Delaunay

    n = node.shape[0]
    B = np.zeros((n, nrhs), dtype=np.complex128)

    xr = [node[:, 0].min(), node[:, 0].max()]
    yr = [node[:, 1].min(), node[:, 1].max()]
    zr = [node[:, 2].min(), node[:, 2].max()]

    # Generate source positions on a grid
    ns = max(1, int(np.ceil(nrhs ** (1 / 3))))
    src_pos = []

    for iz in range(ns):
        for iy in range(ns):
            for ix in range(ns):
                if len(src_pos) >= nrhs:
                    break
                fx = 0.15 + 0.7 * (ix + 0.5) / max(ns, 1)
                fy = 0.15 + 0.7 * (iy + 0.5) / max(ns, 1)
                fz = 0.15 + 0.7 * (iz + 0.5) / max(ns, 1)
                pos = [
                    xr[0] + fx * (xr[1] - xr[0]),
                    yr[0] + fy * (yr[1] - yr[0]),
                    zr[0] + fz * (zr[1] - zr[0]),
                ]
                src_pos.append(pos)
            if len(src_pos) >= nrhs:
                break
        if len(src_pos) >= nrhs:
            break

    src_pos = np.array(src_pos[:nrhs])
    if src_pos.ndim == 1:
        src_pos = src_pos.reshape(1, -1)

    # Find containing elements and compute barycentric coords
    try:
        tri = Delaunay(node)
        simplices = tri.find_simplex(src_pos)

        for k in range(nrhs):
            phase = np.exp(1j * 2 * np.pi * k / max(nrhs, 1))

            if simplices[k] >= 0:
                simplex = tri.simplices[simplices[k]]
                T = tri.transform[simplices[k]]
                bary = T[:3, :3] @ (src_pos[k] - T[3, :])
                bary = np.append(bary, 1 - bary.sum())
                bary = np.maximum(bary, 0)
                bary = bary / bary.sum()
                B[simplex, k] = phase * bary
            else:
                dists = np.sum((node - src_pos[k]) ** 2, axis=1)
                nearest = np.argmin(dists)
                B[nearest, k] = phase
    except Exception:
        for k in range(nrhs):
            phase = np.exp(1j * 2 * np.pi * k / max(nrhs, 1))
            dists = np.sum((node - src_pos[k]) ** 2, axis=1)
            nearest = np.argmin(dists)
            B[nearest, k] = phase

    return B


def create_split_jacobi_precond(A):
    """
    Create split Jacobi preconditioner: M1 = M2 = sqrt(D).
    Matches MATLAB benchmark configuration.
    """
    d = np.array(A.diagonal()).ravel()

    small_thresh = max(np.max(np.abs(d)) * 1e-14, 1e-14)
    small_idx = np.abs(d) < small_thresh
    d[small_idx] = 1.0

    sqrt_d = np.sqrt(d)
    M = sparse.diags(sqrt_d, format="csr")
    return M, M


def create_helmholtz_system(grid_size):
    """Create Helmholtz FEM system matching block_size benchmark."""
    node, elem = meshgrid6(
        np.arange(grid_size), np.arange(grid_size), np.arange(grid_size)
    )
    n = node.shape[0]
    A = assemble_helmholtz_fem(node, elem)
    M1, M2 = create_split_jacobi_precond(A)
    return A, node, elem, M1, M2, n


def _complex_to_real_system(A, B):
    """
    Convert complex system to equivalent real system.

    For complex system (A_r + i*A_i)(x_r + i*x_i) = (b_r + i*b_i):

    [A_r  -A_i] [x_r]   [b_r]
    [A_i   A_r] [x_i] = [b_i]

    Returns real matrix (2n x 2n) and real RHS (2n x nrhs).
    """
    n = A.shape[0]
    A_r = A.real
    A_i = A.imag

    # Build block matrix [A_r, -A_i; A_i, A_r]
    A_real = sparse.bmat([[A_r, -A_i], [A_i, A_r]], format="csc")

    # Build real RHS [b_r; b_i]
    B_r = B.real
    B_i = B.imag
    B_real = np.vstack([B_r, B_i])

    return A_real, B_real


def _real_to_complex_solution(x_real, n):
    """
    Convert real solution back to complex.

    x_real is (2n x nrhs), returns (n x nrhs) complex.
    """
    x_r = x_real[:n, :]
    x_i = x_real[n:, :]
    return x_r + 1j * x_i


def benchmark_direct(A, B, use_pardiso=False, n_runs=2):
    """Benchmark direct solver (PARDISO if requested and available, else SuperLU).

    LU factorization is done once, then multiple RHS are solved.
    This matches how direct solvers are used in practice.

    For complex matrices with PARDISO: converts to real 2x2 block system.
    """
    global HAS_PARDISO, pypardiso_solver

    # Ensure B is 2D
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    nrhs = B.shape[1]
    n = A.shape[0]
    times = []

    # Check if matrix is complex
    is_complex = np.iscomplexobj(A) or np.iscomplexobj(B)

    # Check PARDISO availability if requested
    if use_pardiso and pypardiso_solver is None:
        _try_init_pardiso()

    use_pardiso_now = use_pardiso and HAS_PARDISO
    solver_name = "PARDISO" if use_pardiso_now else "SuperLU"

    for _ in range(n_runs):
        t0 = time.perf_counter()
        if use_pardiso_now:
            try:
                if is_complex:
                    # Convert complex to real 2x2 block system
                    A_r = A.real
                    A_i = A.imag
                    A_real = sparse.bmat([[A_r, -A_i], [A_i, A_r]], format="csr")
                    B_real = np.vstack([B.real, B.imag])

                    x_real = _pardiso_solve_safe(A_real, B_real)
                    if x_real.ndim == 1:
                        x_real = x_real.reshape(-1, 1)
                    x = x_real[:n, :] + 1j * x_real[n:, :]
                else:
                    x = _pardiso_solve_safe(A.tocsr(), B)
            except Exception:
                # PARDISO failed even after retry — fall back to SuperLU
                use_pardiso_now = False
                solver_name = "SuperLU"
                lu = splu(A.tocsc())
                x = np.zeros((n, nrhs), dtype=B.dtype)
                for i in range(nrhs):
                    x[:, i] = lu.solve(B[:, i])

        if not use_pardiso_now:
            # SuperLU: factorize once, solve all RHS
            lu = splu(A.tocsc())
            x = np.zeros((n, nrhs), dtype=B.dtype)
            for i in range(nrhs):
                x[:, i] = lu.solve(B[:, i])
        times.append(time.perf_counter() - t0)

    return np.min(times), x, solver_name


def benchmark_blqmr_block(A, B, M1, M2, block_size=4, tol=1e-6, maxiter=2000, n_runs=2):
    """
    Benchmark BLQMR with fixed block size, processing all RHS in batches.
    Uses Fortran if available, otherwise Native Python.
    """
    n = A.shape[0]
    # Ensure B is 2D
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    nrhs = B.shape[1]
    use_fortran = BLQMR_EXT

    n_full_batches = nrhs // block_size
    remainder = nrhs % block_size

    times = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        total_iters = 0
        x_all = []
        max_flag = 0

        # Process full batches
        for batch in range(n_full_batches):
            start_idx = batch * block_size
            end_idx = start_idx + block_size
            B_batch = B[:, start_idx:end_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if use_fortran:
                    result = _blqmr_fortran(
                        A,
                        B_batch,
                        tol=tol,
                        maxiter=maxiter,
                        x0=None,
                        droptol=0.001,
                        precond_type=3,
                        nblock=1,
                    )
                    x_batch = result.x
                    total_iters += result.iter
                    max_flag = max(max_flag, result.flag)
                else:
                    x_batch, flag, relres, niter, _ = _blqmr_python_impl(
                        A, B_batch, tol=tol, maxiter=maxiter, M1=M1, M2=M2
                    )
                    total_iters += niter
                    max_flag = max(max_flag, flag)

            x_all.append(x_batch.reshape(n, -1))

        # Process remainder
        if remainder > 0:
            B_rem = B[:, n_full_batches * block_size :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if use_fortran:
                    result = _blqmr_fortran(
                        A,
                        B_rem,
                        tol=tol,
                        maxiter=maxiter,
                        x0=None,
                        droptol=0.001,
                        precond_type=3,
                        nblock=1,
                    )
                    x_rem = result.x
                    total_iters += result.iter
                    max_flag = max(max_flag, result.flag)
                else:
                    x_rem, flag, relres, niter, _ = _blqmr_python_impl(
                        A, B_rem, tol=tol, maxiter=maxiter, M1=M1, M2=M2
                    )
                    total_iters += niter
                    max_flag = max(max_flag, flag)
            x_all.append(x_rem.reshape(n, -1))

        x = np.hstack(x_all) if len(x_all) > 1 else x_all[0]
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        times.append(time.perf_counter() - t0)

    n_batches = n_full_batches + (1 if remainder > 0 else 0)
    backend = "Fortran" if use_fortran else "Native"
    return np.min(times), x, total_iters, max_flag, n_batches, backend


def max_residual(A, X, B):
    """Compute maximum relative residual across all RHS."""
    # Ensure X and B are 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    nrhs = B.shape[1]
    return np.max(
        [
            np.linalg.norm(A @ X[:, i] - B[:, i]) / np.linalg.norm(B[:, i])
            for i in range(nrhs)
        ]
    )


def run_sweep(grid_sizes, rhs_counts, tol=1e-6, maxiter=2000):
    """Run benchmark sweep over grid sizes and RHS counts."""

    results = {}

    for grid in grid_sizes:
        print(f"\n{'='*80}")
        print(f"GRID {grid}³")
        print(f"{'='*80}")

        # Create Helmholtz FEM system (matching block_size benchmark)
        A, node, elem, M1, M2, n = create_helmholtz_system(grid)
        print(f"  Matrix: n={n}, nnz={A.nnz}, complex symmetric")

        results[grid] = {"n": n, "nnz": A.nnz, "rhs_results": {}}

        for nrhs in rhs_counts:
            print(f"\n  RHS={nrhs}:")

            # Generate RHS (distributed sources matching block_size benchmark)
            B = create_distributed_sources(node, elem, nrhs)

            # Direct solver (set use_pardiso=True to use PARDISO with real form for complex)
            print(f"    Direct...", end=" ", flush=True)
            t_direct, x_direct, solver_name = benchmark_direct(A, B, use_pardiso=True)
            res_direct = max_residual(A, x_direct, B)
            print(f"{t_direct:.3f}s [{solver_name}] (res={res_direct:.2e})")

            # BLQMR with block size 4
            print(f"    BLQMR-4...", end=" ", flush=True)
            (
                t_blqmr,
                x_blqmr,
                iters_blqmr,
                flag_blqmr,
                n_batches,
                backend,
            ) = benchmark_blqmr_block(
                A, B, M1, M2, block_size=nrhs, tol=tol, maxiter=maxiter
            )
            if flag_blqmr <= 1:
                res_blqmr = max_residual(A, x_blqmr, B)
                print(
                    f"{t_blqmr:.3f}s [{backend}] ({n_batches} batches, {iters_blqmr} iters, res={res_blqmr:.2e})"
                )
            else:
                print(f"FAILED [{backend}] (flag={flag_blqmr})")
                t_blqmr = None

            results[grid]["rhs_results"][nrhs] = {
                "t_direct": t_direct,
                "t_blqmr": t_blqmr,
                "iters_blqmr": iters_blqmr,
                "flag_blqmr": flag_blqmr,
                "direct_solver": solver_name,
                "blqmr_backend": backend,
            }

    return results


def print_summary_table(results, rhs_counts):
    """Print summary table showing crossover points."""

    # Get solver names from first result
    first_grid = list(results.keys())[0]
    first_rhs = list(results[first_grid]["rhs_results"].keys())[0]
    direct_solver = results[first_grid]["rhs_results"][first_rhs]["direct_solver"]
    blqmr_backend = results[first_grid]["rhs_results"][first_rhs]["blqmr_backend"]

    print(f"\n{'='*120}")
    print(f"SUMMARY: {direct_solver} vs {blqmr_backend}-BLQMR-4")
    print(f"{'='*120}")

    # Timing table - Direct
    print(f"\n{direct_solver} times (seconds):")
    print("─" * 100)

    header = f"{'Grid':>12} │"
    for nrhs in rhs_counts:
        header += f" {nrhs:>7} │"
    print(header)
    print("─" * len(header))

    for grid in sorted(results.keys()):
        row = f"{grid}³ (n={results[grid]['n']:>5}) │"
        for nrhs in rhs_counts:
            t = results[grid]["rhs_results"][nrhs]["t_direct"]
            row += f" {t:>7.3f} │"
        print(row)

    # Timing table - BLQMR
    print(f"\n{blqmr_backend}-BLQMR-4 times (seconds):")
    print("─" * 100)

    header = f"{'Grid':>12} │"
    for nrhs in rhs_counts:
        header += f" {nrhs:>7} │"
    print(header)
    print("─" * len(header))

    for grid in sorted(results.keys()):
        row = f"{grid}³ (n={results[grid]['n']:>5}) │"
        for nrhs in rhs_counts:
            t = results[grid]["rhs_results"][nrhs].get("t_blqmr")
            if t is not None:
                row += f" {t:>7.3f} │"
            else:
                row += f" {'FAIL':>7} │"
        print(row)

    # Speedup table
    print(f"\n{'='*120}")
    print(
        f"SPEEDUP: {direct_solver}_time / BLQMR_time (>1.0 = BLQMR wins, marked with *)"
    )
    print(f"{'='*120}")
    print("─" * 100)

    header = f"{'Grid':>12} │"
    for nrhs in rhs_counts:
        header += f" {nrhs:>7} │"
    print(header)
    print("─" * len(header))

    for grid in sorted(results.keys()):
        row = f"{grid}³ (n={results[grid]['n']:>5}) │"
        for nrhs in rhs_counts:
            t_direct = results[grid]["rhs_results"][nrhs]["t_direct"]
            t_blqmr = results[grid]["rhs_results"][nrhs].get("t_blqmr")
            if t_blqmr is not None and t_blqmr > 0:
                speedup = t_direct / t_blqmr
                if speedup >= 1.0:
                    row += f" *{speedup:>5.2f}* │"
                else:
                    row += f"  {speedup:>5.2f}  │"
            else:
                row += f" {'N/A':>7} │"
        print(row)

    # Crossover analysis
    print(f"\n{'='*120}")
    print("CROSSOVER ANALYSIS")
    print(f"{'='*120}")
    print(
        f"\nFor each grid size, find minimum RHS where BLQMR becomes faster than {direct_solver}:\n"
    )

    for grid in sorted(results.keys()):
        n = results[grid]["n"]
        crossover_rhs = None
        crossover_speedup = None

        for nrhs in rhs_counts:
            t_direct = results[grid]["rhs_results"][nrhs]["t_direct"]
            t_blqmr = results[grid]["rhs_results"][nrhs].get("t_blqmr")

            if t_blqmr is not None and t_direct > t_blqmr:
                crossover_rhs = nrhs
                crossover_speedup = t_direct / t_blqmr
                break

        if crossover_rhs is not None:
            print(
                f"  Grid {grid:>2}³ (n={n:>6,}): BLQMR wins at RHS ≥ {crossover_rhs:>3}  (speedup = {crossover_speedup:.2f}x)"
            )
        else:
            # Find best speedup even if < 1
            best_speedup = 0
            best_rhs = None
            for nrhs in rhs_counts:
                t_direct = results[grid]["rhs_results"][nrhs]["t_direct"]
                t_blqmr = results[grid]["rhs_results"][nrhs].get("t_blqmr")
                if t_blqmr is not None and t_blqmr > 0:
                    speedup = t_direct / t_blqmr
                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_rhs = nrhs

            if best_speedup > 0:
                print(
                    f"  Grid {grid:>2}³ (n={n:>6,}): {direct_solver} wins all tested RHS (best BLQMR ratio: {best_speedup:.2f}x at RHS={best_rhs})"
                )
            else:
                print(f"  Grid {grid:>2}³ (n={n:>6,}): No valid BLQMR results")


def main():
    print("=" * 80)
    print("BENCHMARK SWEEP: Grid Size vs RHS Count")
    print("Finding crossover points where BLQMR beats direct solver")
    print("=" * 80)

    print(f"\nBackend Configuration:")
    print(
        f"  Direct solver: PARDISO (complex->real 2x2 block form) if available, else SuperLU"
    )
    print(f"  BLQMR backend: {'Fortran' if BLQMR_EXT else 'Native Python (fallback)'}")
    print(f"  Numba acceleration: {'ENABLED' if HAS_NUMBA else 'DISABLED'}")

    block_size = 0
    tol = 1e-6
    maxiter = 2000

    print(f"\nSolver Configuration:")
    print(f"  BLQMR block size: 0")
    print(f"  Tolerance: {tol}")
    print(f"  Max iterations: {maxiter}")
    print(f"  Preconditioner: split Jacobi (M1 = M2 = sqrt(D))")
    print(f"  Matrix: complex symmetric Helmholtz FEM (tetrahedral mesh)")

    # Sweep parameters
    grid_sizes = [10, 15, 20, 25, 30, 35, 40]
    grid_sizes = [30]
    rhs_counts = [1, 2, 4, 8, 16, 32, 64, 128]

    print(f"\nSweep Parameters:")
    print(f"  Grid sizes: {grid_sizes}")
    print(f"  -> Node counts: {[g**3 for g in grid_sizes]}")
    print(f"  RHS counts: {rhs_counts}")

    results = run_sweep(grid_sizes, rhs_counts, tol=tol, maxiter=maxiter)

    print_summary_table(results, rhs_counts)


if __name__ == "__main__":
    main()
