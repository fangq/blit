"""
BLIT - Block Iterative Sparse Linear Solvers

A Python interface to the BLIT Fortran library for solving sparse linear systems.

Examples
--------
>>> from blit import blqmr_solve
>>> result = blqmr_solve(Ap, Ai, Ax, b)
>>> print(result.x, result.converged)

>>> # With scipy sparse matrices:
>>> from blit import blqmr_scipy
>>> x, flag = blqmr_scipy(A, b)
"""

from .blqmr import (
    blqmr_solve,
    blqmr_solve_multi,
    blqmr_scipy,
    BLQMRResult,
)

__version__ = "1.0.0"
__author__ = "Qianqian Fang"
__email__ = "q.fang@neu.edu"

__all__ = [
    "blqmr_solve",
    "blqmr_solve_multi",
    "blqmr_scipy",
    "BLQMRResult",
]


def test():
    """Run basic tests to verify installation."""
    from .blqmr import _test

    return _test()
