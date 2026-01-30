"""
BLIT - Block Iterative Sparse Linear Solvers

A Python interface to the BLIT Fortran library for solving sparse linear systems.
Falls back to pure-Python implementation when Fortran extension is unavailable.

Examples
--------
>>> from blocksolver import blqmr_solve
>>> result = blqmr_solve(Ap, Ai, Ax, b)
>>> print(result.x, result.converged)

>>> # With scipy sparse matrices:
>>> from blocksolver import blqmr_scipy
>>> x, flag = blqmr_scipy(A, b)

>>> # Direct block QMR with custom preconditioner:
>>> from blocksolver import blqmr, make_preconditioner
>>> M1 = make_preconditioner(A, 'ilu')
>>> x, flag, relres, niter, resv = blqmr(A, b, M1=M1)

>>> # Check which backend is being used:
>>> from blocksolver import BLQMR_EXT
>>> print("Using Fortran" if BLQMR_EXT else "Using pure Python")
"""

from .blqmr import (
    blqmr_solve,
    blqmr_solve_multi,
    blqmr_scipy,
    blqmr,
    BLQMRResult,
    BLQMR_EXT,
    qqr,
    BLQMRWorkspace,
    SparsePreconditioner,
    DensePreconditioner,
    make_preconditioner,
    HAS_NUMBA,
)

__version__ = "0.8.5"
__author__ = "Qianqian Fang"

__all__ = [
    "blqmr_solve",
    "blqmr_solve_multi",
    "blqmr_scipy",
    "blqmr",
    "BLQMRResult",
    "BLQMR_EXT",
    "HAS_NUMBA",
    "qqr",
    "BLQMRWorkspace",
    "SparsePreconditioner",
    "DensePreconditioner",
    "make_preconditioner",
]


def test():
    """Run basic tests to verify installation."""
    from .blqmr import _test

    return _test()


def get_backend_info():
    """Return information about the active backend.

    Returns
    -------
    dict
        Dictionary containing:
        - 'backend': 'binary' or 'native'
        - 'has_fortran': bool
        - 'has_numba': bool (for Python backend acceleration)
    """
    return {
        "backend": "binary" if BLQMR_EXT else "native",
        "has_fortran": BLQMR_EXT,
        "has_numba": HAS_NUMBA,
    }
