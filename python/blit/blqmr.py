"""
High-level Python interface for BLQMR sparse linear solver.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# Import the f2py-generated extension
try:
    from blit import _blqmr
except ImportError:
    try:
        import _blqmr
    except ImportError as e:
        raise ImportError(
            "Could not import _blqmr extension module. "
            "Make sure the package is built with: pip install .\n"
            "Also ensure blit_blqmr_f2py.f90 exists in src/"
        ) from e


@dataclass
class BLQMRResult:
    """Result container for BLQMR solver."""
    x: np.ndarray
    flag: int
    iter: int
    relres: float
    
    @property
    def converged(self) -> bool:
        return self.flag == 0
    
    def __repr__(self) -> str:
        status = "converged" if self.converged else f"flag={self.flag}"
        return f"BLQMRResult({status}, iter={self.iter}, relres={self.relres:.2e})"


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
        Initial guess (currently unused).
    tol : float, default 1e-6
        Convergence tolerance for relative residual.
    maxiter : int, optional
        Maximum iterations. Default is n.
    droptol : float, default 0.001
        Drop tolerance for ILU preconditioner.
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
    # Get dimensions
    n = len(Ap) - 1
    nnz = len(Ax)
    
    # Convert to Fortran-compatible types and ensure contiguous arrays
    # Use Fortran ordering for arrays passed to Fortran
    Ap = np.asfortranarray(Ap, dtype=np.int32)
    Ai = np.asfortranarray(Ai, dtype=np.int32)
    Ax = np.asfortranarray(Ax, dtype=np.float64)
    b = np.asfortranarray(b, dtype=np.float64)
    
    # Validation
    if len(Ai) != nnz:
        raise ValueError(f"Ai length ({len(Ai)}) must match Ax length ({nnz})")
    if len(b) != n:
        raise ValueError(f"b length ({len(b)}) must match matrix size ({n})")
    
    # Convert to 1-based indexing if needed (Fortran convention)
    if zero_based:
        Ap = Ap + 1
        Ai = Ai + 1
    
    if maxiter is None:
        maxiter = n
    dopcond = 1 if use_precond else 0
    
    # Call Fortran solver via f2py
    # Note: f2py expects arguments in exact order matching the .pyf signature
    x, flag, niter, relres = _blqmr.blqmr_solve_real(
        n, nnz, Ap, Ai, Ax, b,
        maxiter, tol, droptol, dopcond
    )
    
    return BLQMRResult(x=x.copy(), flag=int(flag), iter=int(niter), relres=float(relres))


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
    """
    n = len(Ap) - 1
    nnz = len(Ax)
    
    Ap = np.asfortranarray(Ap, dtype=np.int32)
    Ai = np.asfortranarray(Ai, dtype=np.int32)
    Ax = np.asfortranarray(Ax, dtype=np.float64)
    B = np.asfortranarray(B, dtype=np.float64)
    
    if B.ndim == 1:
        B = B.reshape(-1, 1, order='F')
    nrhs = B.shape[1]
    
    if zero_based:
        Ap = Ap + 1
        Ai = Ai + 1
    
    if maxiter is None:
        maxiter = n
    dopcond = 1 if use_precond else 0
    
    X, flag, niter, relres = _blqmr.blqmr_solve_real_multi(
        n, nnz, nrhs, Ap, Ai, Ax, B,
        maxiter, tol, droptol, dopcond
    )
    
    return BLQMRResult(x=X.copy(), flag=int(flag), iter=int(niter), relres=float(relres))


def blqmr_scipy(
    A,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    maxiter: Optional[int] = None,
    M=None,
    **kwargs
) -> Tuple[np.ndarray, int]:
    """
    SciPy-compatible interface for BLQMR solver.
    """
    try:
        from scipy.sparse import csc_matrix
    except ImportError:
        raise ImportError("scipy required for blqmr_scipy")
    
    A_csc = csc_matrix(A)
    Ap = A_csc.indptr.astype(np.int32)
    Ai = A_csc.indices.astype(np.int32)
    Ax = A_csc.data.astype(np.float64)
    
    result = blqmr_solve(Ap, Ai, Ax, b, x0=x0, tol=tol, 
                         maxiter=maxiter, zero_based=True, **kwargs)
    return result.x, result.flag


def _test():
    """Quick test to verify installation."""
    print("BLIT BLQMR Test")
    print("=" * 40)
    
    n, nnz = 5, 12
    # 0-based indexing
    Ap = np.array([0, 2, 5, 9, 10, 12], dtype=np.int32)
    Ai = np.array([0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4], dtype=np.int32)
    Ax = np.array([2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1.], dtype=np.float64)
    b = np.array([8.0, 45.0, -3.0, 3.0, 19.0], dtype=np.float64)
    
    print(f"Matrix: {n}x{n}, nnz={nnz}")
    print(f"Ap: {Ap}")
    print(f"Ai: {Ai}")
    print(f"Ax: {Ax}")
    print(f"b: {b}")
    
    # Debug: Check what f2py expects
    print("\nCalling Fortran solver...")
    
    result = blqmr_solve(Ap, Ai, Ax, b, tol=1e-8, droptol=0.0001)
    
    print(f"\n{result}")
    print(f"Solution: {result.x}")
    
    # Verify
    A = np.zeros((n, n))
    for j in range(n):
        for p in range(Ap[j], Ap[j+1]):
            A[Ai[p], j] = Ax[p]
    res = np.linalg.norm(A @ result.x - b)
    print(f"||Ax - b|| = {res:.2e}")
    
    return result.converged


if __name__ == '__main__':
    _test()
