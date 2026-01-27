"""
BLQMR - Block Quasi-Minimal-Residual sparse linear solver.

This module provides a unified interface that uses the Fortran extension
when available, falling back to a pure-Python implementation otherwise.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spilu
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import warnings

__all__ = [
    "blqmr_solve",
    "blqmr_solve_multi",
    "blqmr_scipy",
    "blqmr",
    "BLQMRResult",
    "BLQMR_EXT",
    "qqr",
    "BLQMRWorkspace",
    "SparsePreconditioner",
    "DensePreconditioner",
    "make_preconditioner",
]

# =============================================================================
# Backend Detection
# =============================================================================

BLQMR_EXT = False
_blqmr = None

try:
    from blocksolver import _blqmr

    BLQMR_EXT = True
except ImportError:
    try:
        import _blqmr

        BLQMR_EXT = True
    except ImportError:
        pass

# Optional Numba acceleration
try:
    from numba import njit

    HAS_NUMBA = True
except (ImportError, Exception) as e:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator if not args or callable(args[0]) else decorator


# =============================================================================
# Result Container
# =============================================================================


@dataclass
class BLQMRResult:
    """Result container for BLQMR solver."""

    x: np.ndarray
    flag: int
    iter: int
    relres: float
    resv: Optional[np.ndarray] = None

    @property
    def converged(self) -> bool:
        return self.flag == 0

    def __repr__(self) -> str:
        status = "converged" if self.converged else f"flag={self.flag}"
        backend = "fortran" if BLQMR_EXT else "python"
        return f"BLQMRResult({status}, iter={self.iter}, relres={self.relres:.2e}, backend={backend})"


# =============================================================================
# Quasi-QR Decomposition
# =============================================================================


@njit(cache=True)
def _qqr_kernel_complex(Q, R, n, m):
    """Numba-accelerated quasi-QR kernel for complex arrays."""
    for j in range(m):
        r_jj_sq = 0.0j
        for i in range(n):
            r_jj_sq += Q[i, j] * Q[i, j]
        r_jj = np.sqrt(r_jj_sq)
        R[j, j] = r_jj
        if abs(r_jj) > 1e-14:
            inv_r_jj = 1.0 / r_jj
            for i in range(n):
                Q[i, j] *= inv_r_jj
            for k in range(j + 1, m):
                dot = 0.0j
                for i in range(n):
                    dot += Q[i, j] * Q[i, k]
                R[j, k] = dot
                for i in range(n):
                    Q[i, k] -= Q[i, j] * dot


@njit(cache=True)
def _qqr_kernel_real(Q, R, n, m):
    """Numba-accelerated quasi-QR kernel for real arrays."""
    for j in range(m):
        r_jj_sq = 0.0
        for i in range(n):
            r_jj_sq += Q[i, j] * Q[i, j]
        r_jj = np.sqrt(r_jj_sq)
        R[j, j] = r_jj
        if abs(r_jj) > 1e-14:
            inv_r_jj = 1.0 / r_jj
            for i in range(n):
                Q[i, j] *= inv_r_jj
            for k in range(j + 1, m):
                dot = 0.0
                for i in range(n):
                    dot += Q[i, j] * Q[i, k]
                R[j, k] = dot
                for i in range(n):
                    Q[i, k] -= Q[i, j] * dot


@njit(cache=True)
def _qqr_kernel_complex(Q, R, n, m):
    """Numba-accelerated quasi-QR kernel for complex arrays."""
    for j in range(m):
        # Quasi inner product: sum(q*q) WITHOUT conjugation
        r_jj_sq = 0.0j
        for i in range(n):
            r_jj_sq += Q[i, j] * Q[i, j]  # No conjugation!
        r_jj = np.sqrt(r_jj_sq)
        R[j, j] = r_jj
        if abs(r_jj) > 1e-14:
            inv_r_jj = 1.0 / r_jj
            for i in range(n):
                Q[i, j] *= inv_r_jj
            for k in range(j + 1, m):
                # Quasi inner product: sum(q_j * q_k) WITHOUT conjugation
                dot = 0.0j
                for i in range(n):
                    dot += Q[i, j] * Q[i, k]  # No conjugation!
                R[j, k] = dot
                for i in range(n):
                    Q[i, k] -= Q[i, j] * dot


@njit(cache=True)
def _qqr_kernel_real(Q, R, n, m):
    """Numba-accelerated quasi-QR kernel for real arrays."""
    for j in range(m):
        r_jj_sq = 0.0
        for i in range(n):
            r_jj_sq += Q[i, j] * Q[i, j]
        r_jj = np.sqrt(r_jj_sq)
        R[j, j] = r_jj
        if abs(r_jj) > 1e-14:
            inv_r_jj = 1.0 / r_jj
            for i in range(n):
                Q[i, j] *= inv_r_jj
            for k in range(j + 1, m):
                dot = 0.0
                for i in range(n):
                    dot += Q[i, j] * Q[i, k]
                R[j, k] = dot
                for i in range(n):
                    Q[i, k] -= Q[i, j] * dot


def qqr(
    A: np.ndarray, tol: float = 0, use_numba: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quasi-QR decomposition using modified Gram-Schmidt with quasi inner product.

    For complex symmetric systems, uses <x,y>_Q = sum(x_k * y_k) without conjugation.

    Parameters
    ----------
    A : ndarray
        Input matrix (n x m)
    tol : float
        Tolerance (unused, for API compatibility)
    use_numba : bool
        If True and Numba available, use JIT-compiled kernel

    Returns
    -------
    Q : ndarray
        Quasi-orthonormal columns (n x m)
    R : ndarray
        Upper triangular matrix (m x m)
    """
    n, m = A.shape
    is_complex = np.iscomplexobj(A)
    dtype = np.complex128 if is_complex else np.float64

    Q = np.ascontiguousarray(A, dtype=dtype)
    R = np.zeros((m, m), dtype=dtype)

    if use_numba and HAS_NUMBA:
        if is_complex:
            _qqr_kernel_complex(Q, R, n, m)
        else:
            _qqr_kernel_real(Q, R, n, m)
    else:
        for j in range(m):
            qj = Q[:, j]
            # CRITICAL FIX: Use sum(qj * qj) NOT np.dot(qj, qj)
            # np.dot conjugates the first argument for complex arrays!
            # Fortran: R(k,k)=dsqrt(sum(Q(:,k)*Q(:,k))) - no conjugation
            r_jj_sq = np.sum(qj * qj)  # Quasi inner product - NO conjugation
            r_jj = np.sqrt(r_jj_sq)
            R[j, j] = r_jj
            if np.abs(r_jj) > 1e-14:
                Q[:, j] *= 1.0 / r_jj
                if j < m - 1:
                    # CRITICAL FIX: Quasi inner product for off-diagonal
                    # Fortran: R(k,j)=sum(Q(:,k)*Q(:,j)) - no conjugation
                    for k in range(j + 1, m):
                        R[j, k] = np.sum(Q[:, j] * Q[:, k])  # NO conjugation
                        Q[:, k] -= R[j, k] * Q[:, j]

    return Q, R


# =============================================================================
# Preconditioner Classes
# =============================================================================


class _ILUPreconditioner:
    """Wrapper for ILU preconditioner to work with blqmr."""

    def __init__(self, ilu_factor):
        self.ilu = ilu_factor
        self.shape = (ilu_factor.shape[0], ilu_factor.shape[1])
        self.dtype = ilu_factor.L.dtype

    def solve(self, b):
        # Convert to real if needed for real ILU
        b_solve = b.real if np.isrealobj(self.ilu.L.data) and np.iscomplexobj(b) else b
        if b_solve.ndim == 1:
            return self.ilu.solve(b_solve)
        else:
            x = np.zeros_like(b_solve)
            for i in range(b_solve.shape[1]):
                x[:, i] = self.ilu.solve(b_solve[:, i])
            return x


class SparsePreconditioner:
    """Efficient sparse preconditioner using LU factorization."""

    __slots__ = ("lu1", "lu2", "is_two_part", "is_ilu1", "is_ilu2")

    def __init__(self, M1, M2=None):
        self.is_two_part = M2 is not None
        self.is_ilu1 = isinstance(M1, (_ILUPreconditioner, _LUPreconditioner))
        self.is_ilu2 = (
            isinstance(M2, (_ILUPreconditioner, _LUPreconditioner))
            if M2 is not None
            else False
        )

        if M1 is not None:
            if self.is_ilu1:
                self.lu1 = M1
            else:
                M1_csc = sparse.csc_matrix(M1) if not sparse.isspmatrix_csc(M1) else M1
                self.lu1 = splu(M1_csc)
        else:
            self.lu1 = None

        if M2 is not None:
            if self.is_ilu2:
                self.lu2 = M2
            else:
                M2_csc = sparse.csc_matrix(M2) if not sparse.isspmatrix_csc(M2) else M2
                self.lu2 = splu(M2_csc)
        else:
            self.lu2 = None

    def solve(self, b: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        if self.lu1 is None:
            return b
        if out is None:
            out = np.empty_like(b)

        # Handle dtype conversion for ILU with real data
        if self.is_ilu1:
            result = self.lu1.solve(b)
            if out.dtype != result.dtype:
                out = np.asarray(out, dtype=result.dtype)
            out[:] = result
        else:
            if b.ndim == 1:
                out[:] = self.lu1.solve(b)
            else:
                for i in range(b.shape[1]):
                    out[:, i] = self.lu1.solve(b[:, i])

        if self.is_two_part:
            if self.is_ilu2:
                out[:] = self.lu2.solve(out)
            else:
                if b.ndim == 1:
                    out[:] = self.lu2.solve(out)
                else:
                    for i in range(b.shape[1]):
                        out[:, i] = self.lu2.solve(out[:, i])
        return out


class DensePreconditioner:
    """Efficient dense preconditioner using LU factorization."""

    __slots__ = ("lu1", "piv1", "lu2", "piv2", "is_two_part")

    def __init__(self, M1: Optional[np.ndarray], M2: Optional[np.ndarray] = None):
        from scipy.linalg import lu_factor

        self.is_two_part = M2 is not None
        if M1 is not None:
            self.lu1, self.piv1 = lu_factor(M1)
        else:
            self.lu1 = self.piv1 = None
        if M2 is not None:
            self.lu2, self.piv2 = lu_factor(M2)
        else:
            self.lu2 = self.piv2 = None

    def solve(self, b: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        from scipy.linalg import lu_solve

        if self.lu1 is None:
            return b
        result = lu_solve((self.lu1, self.piv1), b)
        if self.is_two_part:
            result = lu_solve((self.lu2, self.piv2), result)
        if out is not None:
            out[:] = result
            return out
        return result


# =============================================================================
# BL-QMR Workspace
# =============================================================================


class BLQMRWorkspace:
    """Pre-allocated workspace for BL-QMR iterations."""

    __slots__ = (
        "v",
        "vt",
        "beta",
        "alpha",
        "omega",
        "theta",
        "Qa",
        "Qb",
        "Qc",
        "Qd",
        "zeta",
        "zetat",
        "eta",
        "tau",
        "taot",
        "p",
        "stacked",
        "QQ_full",
        "tmp0",
        "tmp1",
        "tmp2",
        "Av",
        "precond_tmp",
        "n",
        "m",
        "dtype",
    )

    def __init__(self, n: int, m: int, dtype=np.float64):
        self.n, self.m = n, m
        self.dtype = dtype
        self.v = np.zeros((n, m, 3), dtype=dtype)
        self.vt = np.zeros((n, m), dtype=dtype)
        self.beta = np.zeros((m, m, 3), dtype=dtype)
        self.alpha = np.zeros((m, m), dtype=dtype)
        self.omega = np.zeros((m, m, 3), dtype=dtype)
        self.theta = np.zeros((m, m), dtype=dtype)
        self.Qa = np.zeros((m, m, 3), dtype=dtype)
        self.Qb = np.zeros((m, m, 3), dtype=dtype)
        self.Qc = np.zeros((m, m, 3), dtype=dtype)
        self.Qd = np.zeros((m, m, 3), dtype=dtype)
        self.zeta = np.zeros((m, m), dtype=dtype)
        self.zetat = np.zeros((m, m), dtype=dtype)
        self.eta = np.zeros((m, m), dtype=dtype)
        self.tau = np.zeros((m, m), dtype=dtype)
        self.taot = np.zeros((m, m), dtype=dtype)
        self.p = np.zeros((n, m, 3), dtype=dtype)
        self.stacked = np.zeros((2 * m, m), dtype=dtype)
        self.QQ_full = np.zeros((2 * m, 2 * m), dtype=dtype)
        self.tmp0 = np.zeros((m, m), dtype=dtype)
        self.tmp1 = np.zeros((m, m), dtype=dtype)
        self.tmp2 = np.zeros((m, m), dtype=dtype)
        self.Av = np.zeros((n, m), dtype=dtype)
        self.precond_tmp = np.zeros((n, m), dtype=dtype)

    def reset(self):
        self.v.fill(0)
        self.beta.fill(0)
        self.omega.fill(0)
        self.Qa.fill(0)
        self.Qb.fill(0)
        self.Qc.fill(0)
        self.Qd.fill(0)
        self.p.fill(0)
        self.taot.fill(0)


# =============================================================================
# Preconditioner Factory
# =============================================================================


def make_preconditioner(A: sparse.spmatrix, precond_type: str = "diag", **kwargs):
    """
    Create a preconditioner for iterative solvers.

    Parameters
    ----------
    A : sparse matrix
        System matrix
    precond_type : str
        'diag' or 'jacobi': Diagonal (Jacobi) preconditioner
        'ilu' or 'ilu0': Incomplete LU with minimal fill
        'ilut': Incomplete LU with threshold (better quality)
        'lu': Full LU factorization (exact, use as reference)
        'ssor': Symmetric SOR
    **kwargs : dict
        Additional parameters for ILU:
        - drop_tol: Drop tolerance (default: 1e-4 for ilut, 0 for ilu0)
        - fill_factor: Fill factor (default: 10 for ilut, 1 for ilu0)

    Returns
    -------
    M : preconditioner object
        Preconditioner (use as M1 in blqmr)
    """
    if precond_type in ("diag", "jacobi"):
        diag = A.diagonal().copy()
        diag[np.abs(diag) < 1e-14] = 1.0
        return sparse.diags(
            1.0 / diag, format="csr"
        )  # Return inverse for preconditioning!

    elif precond_type == "ilu0":
        # ILU(0) - no fill-in, fast but may be poor quality
        try:
            ilu = spilu(A.tocsc(), drop_tol=0, fill_factor=1)
            return _ILUPreconditioner(ilu)
        except Exception as e:
            warnings.warn(f"ILU(0) factorization failed: {e}, falling back to diagonal")
            return make_preconditioner(A, "diag")

    elif precond_type in ("ilu", "ilut"):
        # ILUT - ILU with threshold, better quality (similar to UMFPACK)
        drop_tol = kwargs.get("drop_tol", 1e-4)
        fill_factor = kwargs.get("fill_factor", 10)
        try:
            ilu = spilu(A.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
            return _ILUPreconditioner(ilu)
        except Exception as e:
            warnings.warn(f"ILUT factorization failed: {e}, trying ILU(0)")
            try:
                ilu = spilu(A.tocsc(), drop_tol=0, fill_factor=1)
                return _ILUPreconditioner(ilu)
            except Exception as e2:
                warnings.warn(f"ILU(0) also failed: {e2}, falling back to diagonal")
                return make_preconditioner(A, "diag")

    elif precond_type == "lu":
        # Full LU - exact factorization (for reference/debugging)
        try:
            lu = splu(A.tocsc())
            return _LUPreconditioner(lu)
        except Exception as e:
            warnings.warn(f"LU factorization failed: {e}, falling back to ILUT")
            return make_preconditioner(A, "ilut")

    elif precond_type == "ssor":
        omega = kwargs.get("omega", 1.0)
        D = sparse.diags(A.diagonal(), format="csr")
        L = sparse.tril(A, k=-1, format="csr")
        return (D + omega * L).tocsr()

    else:
        raise ValueError(f"Unknown preconditioner type: {precond_type}")


class _LUPreconditioner:
    """Wrapper for full LU preconditioner."""

    def __init__(self, lu_factor):
        self.lu = lu_factor
        self.shape = (lu_factor.shape[0], lu_factor.shape[1])
        self.dtype = np.float64  # Assume real for now

    def solve(self, b):
        if b.ndim == 1:
            return self.lu.solve(b)
        else:
            x = np.zeros_like(b)
            for i in range(b.shape[1]):
                x[:, i] = self.lu.solve(b[:, i])
            return x


# =============================================================================
# Pure-Python Block QMR Solver
# =============================================================================


def _blqmr_python_impl(
    A: Union[np.ndarray, sparse.spmatrix],
    B: np.ndarray,
    tol: float = 1e-6,
    maxiter: Optional[int] = None,
    M1=None,
    M2=None,
    x0: Optional[np.ndarray] = None,
    residual: bool = False,
    workspace: Optional[BLQMRWorkspace] = None,
) -> Tuple[np.ndarray, int, float, int, np.ndarray]:
    """Native Python Block QMR implementation (internal)."""
    if B.ndim == 1:
        B = B.reshape(-1, 1)

    n, m = B.shape
    is_complex_input = np.iscomplexobj(A) or np.iscomplexobj(B)
    dtype = np.complex128 if is_complex_input else np.float64

    if maxiter is None:
        maxiter = min(n, 100)

    if (
        workspace is None
        or workspace.n != n
        or workspace.m != m
        or workspace.dtype != dtype
    ):
        ws = BLQMRWorkspace(n, m, dtype)
    else:
        ws = workspace
        ws.reset()

    # Setup preconditioner
    if M1 is not None:
        if isinstance(M1, (_ILUPreconditioner, _LUPreconditioner)):
            precond = SparsePreconditioner(M1, M2)
        elif sparse.issparse(M1):
            precond = SparsePreconditioner(M1, M2)
        elif hasattr(M1, "solve"):
            # Custom preconditioner with .solve() method
            precond = M1  # Use directly
        else:
            precond = DensePreconditioner(M1, M2)
    else:
        precond = None

    if x0 is None:
        x = np.zeros((n, m), dtype=dtype)
    else:
        x = np.asarray(x0, dtype=dtype).reshape(n, m).copy()

    # Initialize indices: Fortran t3=mod(0,3)+1=1 -> Python t3=0
    t3 = 0
    t3n = 2
    t3p = 1

    # Initialize Q matrices (identity)
    ws.Qa[:, :, :] = 0
    ws.Qb[:, :, :] = 0
    ws.Qc[:, :, :] = 0
    ws.Qd[:, :, :] = 0
    ws.Qa[:, :, t3] = np.eye(m, dtype=dtype)
    ws.Qd[:, :, t3n] = np.eye(m, dtype=dtype)
    ws.Qd[:, :, t3] = np.eye(m, dtype=dtype)

    A_is_sparse = sparse.issparse(A)
    if A_is_sparse:
        ws.vt[:] = B - A @ x
    else:
        np.subtract(B, A @ x, out=ws.vt)

    if precond is not None:
        precond.solve(ws.vt, out=ws.vt)
        if np.any(np.isnan(ws.vt)):
            return x, 2, 1.0, 0, np.array([])

    # QQR decomposition
    Q, R = qqr(ws.vt)
    ws.v[:, :, t3p] = Q
    ws.beta[:, :, t3p] = R

    # Compute omega - standard norm WITH conjugation (Hermitian norm)
    # Fortran: omega(i,i,t3p)=sqrt(sum(conjg(v(:,i,t3p))*v(:,i,t3p)))
    for i in range(m):
        col = ws.v[:, i, t3p]
        if is_complex_input:
            ws.omega[i, i, t3p] = np.sqrt(np.sum(np.conj(col) * col).real)
        else:
            ws.omega[i, i, t3p] = np.sqrt(np.sum(col * col))

    # taut = omega * beta
    ws.taot[:] = ws.omega[:, :, t3p] @ ws.beta[:, :, t3p]

    isquasires = not residual
    if isquasires:
        # Fortran: Qres0=maxval(sqrt(sum(abs(conjg(taut)*taut),1))) for complex
        if is_complex_input:
            Qres0 = np.max(np.sqrt(np.sum(np.abs(np.conj(ws.taot) * ws.taot), axis=0)))
        else:
            Qres0 = np.max(np.sqrt(np.sum(ws.taot * ws.taot, axis=0)))
    else:
        omegat = np.zeros((n, m), dtype=dtype)
        for i in range(m):
            if np.abs(ws.omega[i, i, t3p]) > 1e-14:
                omegat[:, i] = ws.v[:, i, t3p] / ws.omega[i, i, t3p]
        if is_complex_input:
            Qres0 = np.max(np.sqrt(np.sum(np.abs(np.conj(ws.vt) * ws.vt), axis=0)))
        else:
            Qres0 = np.max(np.sqrt(np.sum(ws.vt * ws.vt, axis=0)))

    if Qres0 < 1e-16:
        result = x.real if not is_complex_input else x
        return result, 0, 0.0, 0, np.array([0.0])

    flag, resv, Qres1, relres, iter_count = 1, np.zeros(maxiter), -1.0, 1.0, 0

    for k in range(1, maxiter + 1):
        # Index cycling
        t3 = k % 3
        t3p = (k + 1) % 3
        t3n = (k - 1) % 3
        t3nn = (k - 2) % 3

        # tmp = A * v(:,:,t3)
        if A_is_sparse:
            ws.Av[:] = A @ ws.v[:, :, t3]
        else:
            np.matmul(A, ws.v[:, :, t3], out=ws.Av)

        # Apply preconditioner
        if precond is not None:
            precond.solve(ws.Av, out=ws.vt)
            ws.vt[:] = ws.vt - ws.v[:, :, t3n] @ ws.beta[:, :, t3].T
        else:
            ws.vt[:] = ws.Av - ws.v[:, :, t3n] @ ws.beta[:, :, t3].T

        # alpha = v^T * vt (transpose, not conjugate transpose)
        ws.alpha[:] = ws.v[:, :, t3].T @ ws.vt
        ws.vt[:] = ws.vt - ws.v[:, :, t3] @ ws.alpha

        # QQR decomposition
        Q, R = qqr(ws.vt)
        ws.v[:, :, t3p] = Q
        ws.beta[:, :, t3p] = R

        # Compute omega (standard Hermitian norm)
        for i in range(m):
            col = ws.v[:, i, t3p]
            if is_complex_input:
                ws.omega[i, i, t3p] = np.sqrt(np.sum(np.conj(col) * col).real)
            else:
                ws.omega[i, i, t3p] = np.sqrt(np.sum(col * col))

        # Compute intermediate matrices
        ws.tmp0[:] = ws.omega[:, :, t3n] @ ws.beta[:, :, t3].T
        ws.theta[:] = ws.Qb[:, :, t3nn] @ ws.tmp0
        ws.tmp1[:] = ws.Qd[:, :, t3nn] @ ws.tmp0
        ws.tmp2[:] = ws.omega[:, :, t3] @ ws.alpha
        ws.eta[:] = ws.Qa[:, :, t3n] @ ws.tmp1 + ws.Qb[:, :, t3n] @ ws.tmp2
        ws.zetat[:] = ws.Qc[:, :, t3n] @ ws.tmp1 + ws.Qd[:, :, t3n] @ ws.tmp2

        # Build ZZ matrix and do standard QR
        ws.stacked[:m, :] = ws.zetat
        ws.stacked[m:, :] = ws.omega[:, :, t3p] @ ws.beta[:, :, t3p]

        QQ, zeta_full = np.linalg.qr(ws.stacked, mode="complete")
        ws.zeta[:] = zeta_full[:m, :]

        if is_complex_input:
            ws.QQ_full[:] = np.conj(QQ.T)
        else:
            ws.QQ_full[:] = QQ.T

        ws.Qa[:, :, t3] = ws.QQ_full[:m, :m]
        ws.Qb[:, :, t3] = ws.QQ_full[:m, m : 2 * m]
        ws.Qc[:, :, t3] = ws.QQ_full[m : 2 * m, :m]
        ws.Qd[:, :, t3] = ws.QQ_full[m : 2 * m, m : 2 * m]

        # Invert zeta
        try:
            zeta_inv = np.linalg.inv(ws.zeta)
        except np.linalg.LinAlgError:
            zeta_inv = np.linalg.pinv(ws.zeta)

        # Update p, tau, x, taut
        ws.p[:, :, t3] = (
            ws.v[:, :, t3] - ws.p[:, :, t3n] @ ws.eta - ws.p[:, :, t3nn] @ ws.theta
        ) @ zeta_inv
        ws.tau[:] = ws.Qa[:, :, t3] @ ws.taot
        x[:] = x + ws.p[:, :, t3] @ ws.tau
        ws.taot[:] = ws.Qc[:, :, t3] @ ws.taot

        # Compute residual
        if isquasires:
            if is_complex_input:
                Qres = np.max(
                    np.sqrt(np.sum(np.abs(np.conj(ws.taot) * ws.taot), axis=0))
                )
            else:
                Qres = np.max(np.sqrt(np.sum(ws.taot * ws.taot, axis=0)))
        else:
            tmp0_diag = np.zeros((m, m), dtype=dtype)
            for i in range(m):
                if np.abs(ws.omega[i, i, t3p]) > 1e-14:
                    tmp0_diag[i, :] = ws.Qd[:, i, t3] / ws.omega[i, i, t3p]
            if is_complex_input:
                omegat = omegat @ np.conj(ws.Qc[:, :, t3].T) + ws.v[
                    :, :, t3p
                ] @ np.conj(tmp0_diag)
                tmp_res = np.conj(omegat @ ws.taot)
                Qres = np.max(
                    np.sqrt(np.sum(np.abs(np.conj(tmp_res) * tmp_res), axis=0))
                )
            else:
                omegat = omegat @ ws.Qc[:, :, t3].T + ws.v[:, :, t3p] @ tmp0_diag
                tmp_res = omegat @ ws.taot
                Qres = np.max(np.sqrt(np.sum(tmp_res * tmp_res, axis=0)))

        resv[k - 1] = Qres

        if k > 1 and abs(Qres - Qres1) < np.finfo(dtype).eps:
            flag, iter_count = 3, k
            break

        Qres1, relres, iter_count = Qres, Qres / Qres0, k

        if relres <= tol:
            flag = 0
            break

    resv = resv[:iter_count]
    result = x.real if not is_complex_input else x
    return result, flag, relres, iter_count, resv


# =============================================================================
# High-Level Solver Interface
# =============================================================================


def blqmr_solve(
    Ap: np.ndarray,
    Ai: np.ndarray,
    Ax: np.ndarray,
    b: np.ndarray,
    *,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    maxiter: Optional[int] = None,
    droptol: float = 0.001,
    use_precond: bool = True,
    zero_based: bool = True,
) -> BLQMRResult:
    """
    Solve sparse linear system Ax = b using Block QMR algorithm.

    Uses Fortran extension if available, otherwise falls back to pure Python.

    Parameters
    ----------
    Ap : ndarray of int32
        Column pointers for CSC format. Length n+1.
    Ai : ndarray of int32
        Row indices for CSC format. Length nnz.
    Ax : ndarray of float64
        Non-zero values. Length nnz.
    b : ndarray of float64
        Right-hand side vector. Length n.
    x0 : ndarray, optional
        Initial guess.
    tol : float, default 1e-6
        Convergence tolerance for relative residual.
    maxiter : int, optional
        Maximum iterations. Default is n.
    droptol : float, default 0.001
        Drop tolerance for ILU preconditioner (Fortran only).
    use_precond : bool, default True
        Whether to use ILU preconditioning.
    zero_based : bool, default True
        If True, Ap and Ai use 0-based indexing (Python/C convention).
        If False, uses 1-based indexing (Fortran convention).

    Returns
    -------
    BLQMRResult
        Result object containing solution and convergence info.
    """
    n = len(Ap) - 1

    if maxiter is None:
        maxiter = n

    if BLQMR_EXT:
        return _blqmr_solve_fortran(
            Ap,
            Ai,
            Ax,
            b,
            x0=x0,
            tol=tol,
            maxiter=maxiter,
            droptol=droptol,
            use_precond=use_precond,
            zero_based=zero_based,
        )
    else:
        return _blqmr_solve_native_csc(
            Ap,
            Ai,
            Ax,
            b,
            x0=x0,
            tol=tol,
            maxiter=maxiter,
            use_precond=use_precond,
            zero_based=zero_based,
        )


def _blqmr_solve_fortran(
    Ap, Ai, Ax, b, *, x0, tol, maxiter, droptol, use_precond, zero_based
) -> BLQMRResult:
    """Fortran backend for blqmr_solve."""
    n = len(Ap) - 1
    nnz = len(Ax)

    Ap = np.asfortranarray(Ap, dtype=np.int32)
    Ai = np.asfortranarray(Ai, dtype=np.int32)
    Ax = np.asfortranarray(Ax, dtype=np.float64)
    b = np.asfortranarray(b, dtype=np.float64)

    if len(Ai) != nnz:
        raise ValueError(f"Ai length ({len(Ai)}) must match Ax length ({nnz})")
    if len(b) != n:
        raise ValueError(f"b length ({len(b)}) must match matrix size ({n})")

    if zero_based:
        Ap = Ap + 1
        Ai = Ai + 1

    dopcond = 1 if use_precond else 0

    x, flag, niter, relres = _blqmr.blqmr_solve_real(
        n, nnz, Ap, Ai, Ax, b, maxiter, tol, droptol, dopcond
    )

    return BLQMRResult(
        x=x.copy(), flag=int(flag), iter=int(niter), relres=float(relres)
    )


def _blqmr_solve_native_csc(
    Ap, Ai, Ax, b, *, x0, tol, maxiter, use_precond, zero_based
) -> BLQMRResult:
    """Native Python backend for blqmr_solve with CSC input."""
    n = len(Ap) - 1

    if not zero_based:
        Ap = Ap - 1
        Ai = Ai - 1

    A = sparse.csc_matrix((Ax, Ai, Ap), shape=(n, n))

    M1 = None
    if use_precond:
        try:
            M1 = make_preconditioner(A, "ilu")
        except Exception:
            try:
                M1 = make_preconditioner(A, "diag")  # FIX: Changed A_sp to A
            except Exception:
                M1 = None  # Fall back to no preconditioning

    x, flag, relres, niter, resv = _blqmr_python_impl(
        A, b, tol=tol, maxiter=maxiter, M1=M1, x0=x0
    )

    if x.ndim > 1:
        x = x.ravel()

    return BLQMRResult(x=x, flag=flag, iter=niter, relres=relres, resv=resv)


def blqmr_solve_multi(
    Ap: np.ndarray,
    Ai: np.ndarray,
    Ax: np.ndarray,
    B: np.ndarray,
    *,
    tol: float = 1e-6,
    maxiter: Optional[int] = None,
    droptol: float = 0.001,
    use_precond: bool = True,
    zero_based: bool = True,
) -> BLQMRResult:
    """
    Solve sparse linear system AX = B with multiple right-hand sides.

    Uses Fortran extension if available, otherwise falls back to pure Python.
    """
    n = len(Ap) - 1

    if maxiter is None:
        maxiter = n

    if BLQMR_EXT:
        return _blqmr_solve_multi_fortran(
            Ap,
            Ai,
            Ax,
            B,
            tol=tol,
            maxiter=maxiter,
            droptol=droptol,
            use_precond=use_precond,
            zero_based=zero_based,
        )
    else:
        return _blqmr_solve_multi_native(
            Ap,
            Ai,
            Ax,
            B,
            tol=tol,
            maxiter=maxiter,
            use_precond=use_precond,
            zero_based=zero_based,
        )


def _blqmr_solve_multi_fortran(
    Ap, Ai, Ax, B, *, tol, maxiter, droptol, use_precond, zero_based
) -> BLQMRResult:
    """Fortran backend for blqmr_solve_multi."""
    n = len(Ap) - 1
    nnz = len(Ax)

    Ap = np.asfortranarray(Ap, dtype=np.int32)
    Ai = np.asfortranarray(Ai, dtype=np.int32)
    Ax = np.asfortranarray(Ax, dtype=np.float64)
    B = np.asfortranarray(B, dtype=np.float64)

    if B.ndim == 1:
        B = B.reshape(-1, 1, order="F")
    nrhs = B.shape[1]

    if zero_based:
        Ap = Ap + 1
        Ai = Ai + 1

    dopcond = 1 if use_precond else 0

    X, flag, niter, relres = _blqmr.blqmr_solve_real_multi(
        n, nnz, nrhs, Ap, Ai, Ax, B, maxiter, tol, droptol, dopcond
    )

    return BLQMRResult(
        x=X.copy(), flag=int(flag), iter=int(niter), relres=float(relres)
    )


def _blqmr_solve_multi_native(
    Ap, Ai, Ax, B, *, tol, maxiter, use_precond, zero_based
) -> BLQMRResult:
    """Native Python backend for blqmr_solve_multi."""
    n = len(Ap) - 1

    if not zero_based:
        Ap = Ap - 1
        Ai = Ai - 1

    A = sparse.csc_matrix((Ax, Ai, Ap), shape=(n, n))

    M1 = None
    if use_precond:
        try:
            M1 = make_preconditioner(A, "ilu")
        except Exception:
            try:
                M1 = make_preconditioner(A, "diag")  # FIX: Changed A_sp to A
            except Exception:
                M1 = None  # Fall back to no preconditioning

    if B.ndim == 1:
        B = B.reshape(-1, 1)

    x, flag, relres, niter, resv = _blqmr_python_impl(
        A, B, tol=tol, maxiter=maxiter, M1=M1
    )

    return BLQMRResult(x=x, flag=flag, iter=niter, relres=relres, resv=resv)


def blqmr_scipy(
    A,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    maxiter: Optional[int] = None,
    M=None,
    **kwargs,
) -> Tuple[np.ndarray, int]:
    """
    SciPy-compatible interface for BLQMR solver.

    Parameters
    ----------
    A : sparse matrix or ndarray
        System matrix
    b : ndarray
        Right-hand side vector
    x0 : ndarray, optional
        Initial guess
    tol : float
        Convergence tolerance
    maxiter : int, optional
        Maximum iterations
    M : preconditioner, optional
        Preconditioner (used as M1 for Python backend)
    **kwargs
        Additional arguments passed to blqmr()

    Returns
    -------
    x : ndarray
        Solution vector
    flag : int
        Convergence flag (0 = converged)
    """
    result = blqmr(A, b, x0=x0, tol=tol, maxiter=maxiter, M1=M, **kwargs)
    return result.x, result.flag


def blqmr(
    A: Union[np.ndarray, sparse.spmatrix],
    B: np.ndarray,
    tol: float = 1e-6,
    maxiter: Optional[int] = None,
    M1=None,
    M2=None,
    x0: Optional[np.ndarray] = None,
    residual: bool = False,
    workspace: Optional[BLQMRWorkspace] = None,
    droptol: float = 0.001,
    use_precond: bool = True,
) -> BLQMRResult:
    """
    Block Quasi-Minimal-Residual (BL-QMR) solver - main interface.

    Uses Fortran extension if available, otherwise falls back to pure Python.

    Parameters
    ----------
    A : ndarray or sparse matrix
        Symmetric n x n matrix (can be complex)
    B : ndarray
        Right-hand side vector/matrix (n,) or (n x m)
    tol : float
        Convergence tolerance (default: 1e-6)
    maxiter : int, optional
        Maximum iterations (default: n for Fortran, min(n, 20) for Python)
    M1, M2 : preconditioner, optional
        Preconditioner M = M1 @ M2 (Python backend only)
    x0 : ndarray, optional
        Initial guess
    residual : bool
        If True, use true residual for convergence (Python backend only)
    workspace : BLQMRWorkspace, optional
        Pre-allocated workspace (Python backend only)
    droptol : float, default 0.001
        Drop tolerance for ILU preconditioner (Fortran backend only)
    use_precond : bool, default True
        Whether to use ILU preconditioning (Fortran backend only)

    Returns
    -------
    BLQMRResult
        Result object containing:
        - x: Solution array
        - flag: 0 = converged, 1 = max iterations, 2 = preconditioner singular, 3 = stagnated
        - iter: Number of iterations
        - relres: Final relative residual
        - resv: Residual history (Python backend only)
    """
    if BLQMR_EXT:
        return _blqmr_fortran(
            A,
            B,
            tol=tol,
            maxiter=maxiter,
            x0=x0,
            droptol=droptol,
            use_precond=use_precond,
        )
    else:
        return _blqmr_native(
            A,
            B,
            tol=tol,
            maxiter=maxiter,
            M1=M1,
            M2=M2,
            x0=x0,
            residual=residual,
            workspace=workspace,
            use_precond=use_precond,
        )


def _blqmr_fortran(
    A: Union[np.ndarray, sparse.spmatrix],
    B: np.ndarray,
    *,
    tol: float,
    maxiter: Optional[int],
    x0: Optional[np.ndarray],
    droptol: float,
    use_precond: bool,
) -> BLQMRResult:
    """Fortran backend for blqmr()."""
    A_csc = sparse.csc_matrix(A)

    # CRITICAL: Sort indices for UMFPACK compatibility
    if not A_csc.has_sorted_indices:
        A_csc.sort_indices()

    Ap = A_csc.indptr.astype(np.int32)
    Ai = A_csc.indices.astype(np.int32)

    n = A_csc.shape[0]
    nnz = A_csc.nnz

    if maxiter is None:
        maxiter = n

    # Convert to Fortran format (1-based indexing)
    Ap_f = np.asfortranarray(Ap + 1, dtype=np.int32)
    Ai_f = np.asfortranarray(Ai + 1, dtype=np.int32)

    dopcond = 1 if use_precond else 0

    # Check if complex
    is_complex = np.iscomplexobj(A) or np.iscomplexobj(B)

    if is_complex:
        # Complex path
        Ax_f = np.asfortranarray(A_csc.data, dtype=np.complex128)

        if B.ndim == 1 or (B.ndim == 2 and B.shape[1] == 1):
            # Single RHS
            b_f = np.asfortranarray(B.ravel(), dtype=np.complex128)
            x, flag, niter, relres = _blqmr.blqmr_solve_complex(
                n, nnz, Ap_f, Ai_f, Ax_f, b_f, maxiter, tol, droptol, dopcond
            )
            return BLQMRResult(
                x=x.copy(), flag=int(flag), iter=int(niter), relres=float(relres)
            )
        else:
            # Multiple RHS - use block method
            B_f = np.asfortranarray(B, dtype=np.complex128)
            nrhs = B_f.shape[1]
            X, flag, niter, relres = _blqmr.blqmr_solve_complex_multi(
                n, nnz, nrhs, Ap_f, Ai_f, Ax_f, B_f, maxiter, tol, droptol, dopcond
            )
            return BLQMRResult(
                x=X.copy(), flag=int(flag), iter=int(niter), relres=float(relres)
            )
    else:
        # Real path
        Ax_f = np.asfortranarray(A_csc.data, dtype=np.float64)

        if B.ndim == 1 or (B.ndim == 2 and B.shape[1] == 1):
            # Single RHS
            b_f = np.asfortranarray(B.ravel(), dtype=np.float64)
            x, flag, niter, relres = _blqmr.blqmr_solve_real(
                n, nnz, Ap_f, Ai_f, Ax_f, b_f, maxiter, tol, droptol, dopcond
            )
            return BLQMRResult(
                x=x.copy(), flag=int(flag), iter=int(niter), relres=float(relres)
            )
        else:
            # Multiple RHS - use block method
            B_f = np.asfortranarray(B, dtype=np.float64)
            nrhs = B_f.shape[1]
            X, flag, niter, relres = _blqmr.blqmr_solve_real_multi(
                n, nnz, nrhs, Ap_f, Ai_f, Ax_f, B_f, maxiter, tol, droptol, dopcond
            )
            return BLQMRResult(
                x=X.copy(), flag=int(flag), iter=int(niter), relres=float(relres)
            )


def _blqmr_native(
    A: Union[np.ndarray, sparse.spmatrix],
    B: np.ndarray,
    *,
    tol: float,
    maxiter: Optional[int],
    M1,
    M2,
    x0: Optional[np.ndarray],
    residual: bool,
    workspace: Optional[BLQMRWorkspace],
    use_precond: bool,
) -> BLQMRResult:
    """Native Python backend for blqmr()."""
    # Auto-create preconditioner if requested and not provided
    if use_precond and M1 is None:
        A_sp = sparse.csc_matrix(A) if not sparse.issparse(A) else A
        try:
            M1 = make_preconditioner(A_sp, "ilu")
        except Exception:
            try:
                M1 = make_preconditioner(A_sp, "diag")
            except Exception:
                M1 = None  # Fall back to no preconditioning

    x, flag, relres, niter, resv = _blqmr_python_impl(
        A,
        B,
        tol=tol,
        maxiter=maxiter,
        M1=M1,
        M2=M2,
        x0=x0,
        residual=residual,
        workspace=workspace,
    )

    # Flatten x if single RHS
    if x.ndim > 1 and x.shape[1] == 1:
        x = x.ravel()

    return BLQMRResult(x=x, flag=flag, iter=niter, relres=relres, resv=resv)


# =============================================================================
# Test Function
# =============================================================================


def _test():
    """Quick test to verify installation."""
    print("BLIT BLQMR Test")
    print("=" * 40)
    print(f"Fortran backend available: {BLQMR_EXT}")
    print(f"Numba acceleration available: {HAS_NUMBA}")
    print(f"Using backend: {'Fortran' if BLQMR_EXT else 'Pure Python'}")
    print()

    # Build test matrix from CSC components
    n = 5
    Ap = np.array([0, 2, 5, 9, 10, 12], dtype=np.int32)
    Ai = np.array([0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4], dtype=np.int32)
    Ax = np.array(
        [2.0, 3.0, 3.0, -1.0, 4.0, 4.0, -3.0, 1.0, 2.0, 2.0, 6.0, 1.0], dtype=np.float64
    )
    b = np.array([8.0, 45.0, -3.0, 3.0, 19.0], dtype=np.float64)

    # Create sparse matrix
    A = sparse.csc_matrix((Ax, Ai, Ap), shape=(n, n))

    print(f"Matrix: {n}x{n}, nnz={len(Ax)}")
    print(f"b: {b}")
    print("\nCalling blqmr()...")

    result = blqmr(A, b, tol=1e-8)

    print(f"\n{result}")
    print(f"Solution: {result.x}")

    # Verify
    res = np.linalg.norm(A @ result.x - b)
    print(f"||Ax - b|| = {res:.2e}")

    return result.converged


if __name__ == "__main__":
    _test()
