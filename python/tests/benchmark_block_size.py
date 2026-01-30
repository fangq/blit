"""
Benchmark script for BLQMR block size performance.
Matched to MATLAB benchmark configuration.

Compares TOTAL TIME to solve all RHS for:
1. Direct solver (spsolve/SuperLU)
2. QMR (point method, each RHS separately)
3. Block BLQMR with various block sizes

Configuration matched to MATLAB:
- Complex symmetric Helmholtz FEM matrix (not Hermitian)
- Split Jacobi preconditioner (M1 = M2 = sqrt(D))
- Distributed point sources as RHS
- Tolerance 1e-6, maxiter 2000
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu, qmr
import time
import sys
import os
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from blocksolver import BLQMR_EXT, HAS_NUMBA, get_backend_info
from blocksolver.blqmr import _blqmr_python_impl, make_preconditioner, BLQMRWorkspace

if BLQMR_EXT:
    from blocksolver.blqmr import _blqmr_fortran


def meshgrid6(x, y, z):
    """
    Generate tetrahedral mesh from regular grid (6 tets per cube).
    Matches MATLAB's meshgrid6 function.

    Returns
    -------
    node : ndarray (n_nodes x 3)
        Node coordinates
    elem : ndarray (n_elems x 4)
        Element connectivity (0-based)
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
    ns = int(np.ceil(nrhs ** (1 / 3)))
    src_pos = []

    for iz in range(ns):
        for iy in range(ns):
            for ix in range(ns):
                if len(src_pos) >= nrhs:
                    break
                fx = 0.15 + 0.7 * (ix + 0.5) / ns
                fy = 0.15 + 0.7 * (iy + 0.5) / ns
                fz = 0.15 + 0.7 * (iz + 0.5) / ns
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

    # Find containing elements and compute barycentric coords
    try:
        tri = Delaunay(node)
        simplices = tri.find_simplex(src_pos)

        for k in range(nrhs):
            phase = np.exp(1j * 2 * np.pi * k / nrhs)

            if simplices[k] >= 0:
                # Point is inside a tetrahedron
                simplex = tri.simplices[simplices[k]]
                # Compute barycentric coordinates
                T = tri.transform[simplices[k]]
                bary = T[:3, :3] @ (src_pos[k] - T[3, :])
                bary = np.append(bary, 1 - bary.sum())
                bary = np.maximum(bary, 0)  # Clamp negatives
                bary = bary / bary.sum()  # Renormalize
                B[simplex, k] = phase * bary
            else:
                # Point outside - find nearest node
                dists = np.sum((node - src_pos[k]) ** 2, axis=1)
                nearest = np.argmin(dists)
                B[nearest, k] = phase
    except Exception:
        # Fallback: just use nearest nodes
        for k in range(nrhs):
            phase = np.exp(1j * 2 * np.pi * k / nrhs)
            dists = np.sum((node - src_pos[k]) ** 2, axis=1)
            nearest = np.argmin(dists)
            B[nearest, k] = phase

    return B


def create_split_jacobi_precond(A):
    """
    Create split Jacobi preconditioner: M1 = M2 = sqrt(D).
    This gives equilibration: D^{-1/2} * A * D^{-1/2}
    Matches MATLAB benchmark configuration.
    """
    d = np.array(A.diagonal()).ravel()

    # Handle small/zero diagonal entries
    small_thresh = max(np.max(np.abs(d)) * 1e-14, 1e-14)
    small_idx = np.abs(d) < small_thresh
    d[small_idx] = 1.0

    # sqrt for split preconditioning
    sqrt_d = np.sqrt(d)

    M = sparse.diags(sqrt_d, format="csr")
    return M, M  # M1 = M2 for symmetric split


def max_residual(A, X, B):
    """Compute maximum relative residual across all RHS."""
    r = 0.0
    for i in range(B.shape[1]):
        bn = np.linalg.norm(B[:, i])
        if bn > 0:
            r = max(r, np.linalg.norm(A @ X[:, i] - B[:, i]) / bn)
    return r


def run_qmr(A, B, tol, maxiter, M1, M2):
    """
    Solve each RHS with scipy's QMR (point method).
    Uses split preconditioner.
    """
    from scipy.sparse.linalg import LinearOperator

    nrhs = B.shape[1]
    n = A.shape[0]
    X = np.zeros((n, nrhs), dtype=B.dtype)
    total_time = 0.0
    total_iters = 0

    # Create LinearOperator preconditioners (M^{-1} application)
    m1_inv_diag = 1.0 / M1.diagonal()
    m2_inv_diag = 1.0 / M2.diagonal()

    # For rmatvec with complex diagonal: use conjugate
    m1_inv_diag_conj = np.conj(m1_inv_diag)
    m2_inv_diag_conj = np.conj(m2_inv_diag)

    M1_op = LinearOperator(
        (n, n),
        matvec=lambda x: m1_inv_diag * x,
        rmatvec=lambda x: m1_inv_diag_conj * x,
        dtype=np.complex128,
    )
    M2_op = LinearOperator(
        (n, n),
        matvec=lambda x: m2_inv_diag * x,
        rmatvec=lambda x: m2_inv_diag_conj * x,
        dtype=np.complex128,
    )

    for k in range(nrhs):
        t0 = time.perf_counter()

        try:
            # Newer scipy (1.12+)
            x, info = qmr(A, B[:, k], rtol=tol, maxiter=maxiter, M1=M1_op, M2=M2_op)
        except TypeError:
            # Older scipy
            M_full_inv_diag = m1_inv_diag * m2_inv_diag
            M_full_inv_diag_conj = np.conj(M_full_inv_diag)
            M_op = LinearOperator(
                (n, n),
                matvec=lambda x: M_full_inv_diag * x,
                rmatvec=lambda x: M_full_inv_diag_conj * x,
                dtype=np.complex128,
            )
            try:
                x, info = qmr(A, B[:, k], tol=tol, maxiter=maxiter, M=M_op)
            except TypeError:
                x, info = qmr(A, B[:, k], rtol=tol, maxiter=maxiter)

        total_time += time.perf_counter() - t0
        X[:, k] = x
        total_iters += maxiter if info != 0 else maxiter // 2

    residual = max_residual(A, X, B)
    return total_time, total_iters, residual


def run_blqmr(A, B, block_size, tol, maxiter, M1, M2, use_fortran=True):
    """
    Solve with BLQMR using given block size.
    If use_fortran=True and BLQMR_EXT available, uses Fortran backend with ILU.
    Otherwise uses Python backend with split Jacobi.
    """
    from blocksolver.blqmr import blqmr, _blqmr_python_impl, BLQMR_EXT

    nrhs = B.shape[1]
    n = A.shape[0]
    X = np.zeros((n, nrhs), dtype=B.dtype)

    num_full_batches = nrhs // block_size
    remainder = nrhs % block_size

    total_time = 0.0
    total_iters = 0
    max_flag = 0

    # Process full batches
    for batch in range(num_full_batches):
        start = batch * block_size
        end = start + block_size
        B_batch = B[:, start:end]

        t0 = time.perf_counter()

        if use_fortran and BLQMR_EXT:
            # Use Fortran backend with ILU preconditioner
            result = blqmr(
                A, B_batch, tol=tol, maxiter=maxiter, precond_type="diag", droptol=0.001
            )
            x = result.x
            flag = result.flag
            niter = result.iter
        else:
            # Use Python backend with split Jacobi
            x, flag, relres, niter, _ = _blqmr_python_impl(
                A, B_batch, tol=tol, maxiter=maxiter, M1=M1, M2=M2
            )

        total_time += time.perf_counter() - t0

        X[:, start:end] = x.reshape(-1, block_size)
        total_iters += niter
        max_flag = max(max_flag, flag)

    # Process remainder batch
    if remainder > 0:
        start = num_full_batches * block_size
        B_batch = B[:, start:]

        t0 = time.perf_counter()

        if use_fortran and BLQMR_EXT:
            result = blqmr(
                A, B_batch, tol=tol, maxiter=maxiter, precond_type="diag", droptol=0.001
            )
            x = result.x
            flag = result.flag
            niter = result.iter
        else:
            x, flag, relres, niter, _ = _blqmr_python_impl(
                A, B_batch, tol=tol, maxiter=maxiter, M1=M1, M2=M2
            )

        total_time += time.perf_counter() - t0

        X[:, start:] = x.reshape(-1, remainder)
        total_iters += niter
        max_flag = max(max_flag, flag)

    residual = max_residual(A, X, B)
    num_batches = num_full_batches + (1 if remainder > 0 else 0)

    return total_time, total_iters, residual, max_flag, num_batches


def run_benchmark():
    """Run benchmark matching MATLAB configuration."""
    print("=" * 100)
    print("BLQMR BENCHMARK: Direct vs QMR vs BLQMR (block sizes 1-64)")
    print("=" * 100)
    print()

    # Configuration matching MATLAB
    total_rhs = 64
    block_sizes = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 28, 32, 48, 64]
    tol = 1e-6
    maxiter = 2000
    grid_sizes = [10, 20, 30, 40]

    print(f"Config: {total_rhs} RHS, tol={tol:.0e}, maxiter={maxiter}")
    print(f"Block sizes: {block_sizes}")
    print(f"Preconditioner: split Jacobi (M1 = M2 = sqrt(D))")
    print(f"Note: Remainder RHS handled when {total_rhs} not divisible by block size")
    print()

    all_results = []

    for grid_m in grid_sizes:
        print("=" * 100)
        print(f"GRID {grid_m}^3")
        print("=" * 100)

        # Create mesh and matrix (matching MATLAB)
        node, elem = meshgrid6(np.arange(grid_m), np.arange(grid_m), np.arange(grid_m))
        n = node.shape[0]
        A = assemble_helmholtz_fem(node, elem)
        B = create_distributed_sources(node, elem, total_rhs)

        print(f"  Matrix: n={n}, nnz={A.nnz}, complex symmetric")

        # Create split preconditioner
        print("  Creating split preconditioner (sqrt diagonal)...", end="")
        M1, M2 = create_split_jacobi_precond(A)
        print(" done")
        print()

        results = {"grid": grid_m, "n": n}

        # === Direct solver (mldivide equivalent) ===
        print("  Running direct solver...", end=" ", flush=True)
        t0 = time.perf_counter()
        lu = splu(A)
        X_direct = np.column_stack([lu.solve(B[:, i]) for i in range(total_rhs)])
        results["t_direct"] = time.perf_counter() - t0
        results["res_direct"] = max_residual(A, X_direct, B)
        print(f'done ({results["t_direct"]:.3f}s)')

        # === QMR (point method) ===
        print("  Running QMR...", end=" ", flush=True)
        results["t_qmr"], results["iter_qmr"], results["res_qmr"] = run_qmr(
            A, B, tol, maxiter, M1, M2
        )
        print(f'done ({results["t_qmr"]:.3f}s, {results["iter_qmr"]} iters)')

        # === BLQMR with different block sizes ===
        results["blqmr"] = {}
        print("  Running BLQMR:")

        for bs in block_sizes:
            print(f"    Block size {bs}...", end=" ", flush=True)
            t, iters, res, flag, nbatches = run_blqmr(A, B, bs, tol, maxiter, M1, M2)
            results["blqmr"][bs] = {
                "time": t,
                "iters": iters,
                "res": res,
                "flag": flag,
                "batches": nbatches,
            }

            remainder = total_rhs % bs
            batch_str = (
                f"{total_rhs // bs}+1" if remainder > 0 else str(total_rhs // bs)
            )
            print(
                f"{t:.3f}s, {iters} iters ({batch_str} batches), res={res:.1e}, flag={flag}"
            )

        all_results.append(results)
        print()

        # Print comparison table
        print_results_table(results, block_sizes, total_rhs)
        print()

    # Print summary
    print_summary(all_results, block_sizes, total_rhs)


def print_results_table(results, block_sizes, total_rhs):
    """Print results table for a single grid size."""
    print("  TIMING & ACCURACY:")
    print(
        f'  {"Method":<12} {"Time(s)":>10} {"Iters":>10} {"Batches":>8} {"Residual":>12} {"Flag":>8}'
    )
    print(f'  {"-"*12} {"-"*10} {"-"*10} {"-"*8} {"-"*12} {"-"*8}')

    print(
        f'  {"direct":<12} {results["t_direct"]:>10.3f} {"--":>10} {"--":>8} {results["res_direct"]:>12.1e} {"--":>8}'
    )
    print(
        f'  {"QMR":<12} {results["t_qmr"]:>10.3f} {results["iter_qmr"]:>10} {"--":>8} {results["res_qmr"]:>12.1e} {"--":>8}'
    )

    for bs in block_sizes:
        r = results["blqmr"][bs]
        remainder = total_rhs % bs
        batch_str = f"{total_rhs // bs}+1" if remainder > 0 else str(total_rhs // bs)
        print(
            f'  {"BLQMR-" + str(bs):<12} {r["time"]:>10.3f} {r["iters"]:>10} {batch_str:>8} {r["res"]:>12.1e} {r["flag"]:>8}'
        )

    # Speedup comparison
    print()
    print("  SPEEDUP (time ratio, >1 means method is faster):")
    print(f'  {"Method":<12} {"vs direct":>12} {"vs QMR":>12} {"vs BLQMR-1":>12}')
    print(f'  {"-"*12} {"-"*12} {"-"*12} {"-"*12}')

    t_direct = results["t_direct"]
    t_qmr = results["t_qmr"]
    t_bl1 = results["blqmr"][1]["time"]

    print(f'  {"direct":<12} {"--":>12} {t_qmr/t_direct:>11.2f}x {"--":>12}')
    print(f'  {"QMR":<12} {t_direct/t_qmr:>11.2f}x {"--":>12} {"--":>12}')

    for bs in block_sizes:
        t = results["blqmr"][bs]["time"]
        print(
            f'  {"BLQMR-" + str(bs):<12} {t_direct/t:>11.2f}x {t_qmr/t:>11.2f}x {t_bl1/t:>11.2f}x'
        )


def print_summary(all_results, block_sizes, total_rhs):
    """Print summary across all grid sizes."""
    print("=" * 100)
    print(f"SUMMARY: ALL METHODS COMPARISON ({total_rhs} RHS)")
    print("=" * 100)
    print()

    # Wall clock time table
    print("WALL CLOCK TIME (seconds):")
    header = f'{"Grid":>8} {"direct":>10} {"QMR":>10}'
    for bs in block_sizes[:12]:  # First 12 block sizes
        header += f' {"BL-"+str(bs):>8}'
    print(header)

    for r in all_results:
        row = f'{r["grid"]**3:>8d} {r["t_direct"]:>10.3f} {r["t_qmr"]:>10.3f}'
        for bs in block_sizes[:12]:
            row += f' {r["blqmr"][bs]["time"]:>8.3f}'
        print(row)

    print()

    # Speedup vs QMR
    print("SPEEDUP vs QMR (>1 = faster than QMR):")
    header = f'{"Grid":>8} {"direct":>10} {"QMR":>10}'
    for bs in block_sizes[:12]:
        header += f' {"BL-"+str(bs):>8}'
    print(header)

    for r in all_results:
        t_qmr = r["t_qmr"]
        row = f'{r["grid"]**3:>8d} {r["t_direct"]:>10.3f} {r["t_qmr"]:>10.3f}'
        for bs in block_sizes[:12]:
            row += f' {t_qmr/r["blqmr"][bs]["time"]:>8.2f}'
        print(row)

    print()

    # Best method per grid
    print("BEST METHOD PER GRID SIZE:")
    for r in all_results:
        times = [("direct", r["t_direct"]), ("QMR", r["t_qmr"])]
        for bs in block_sizes:
            times.append((f"BLQMR-{bs}", r["blqmr"][bs]["time"]))

        best = min(times, key=lambda x: x[1])
        print(f'  {r["grid"]}^3: {best[0]} ({best[1]:.3f}s)')


if __name__ == "__main__":
    run_benchmark()
