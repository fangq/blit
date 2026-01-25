"""
Tests for BLIT BLQMR solver.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from blocksolver import blqmr_solve, blqmr_solve_multi, BLQMRResult


# Test matrix from Fortran test program
@pytest.fixture
def test_system():
    """5x5 sparse test system."""
    n, nnz = 5, 12
    # 0-based indexing (Python convention)
    Ap = np.array([0, 2, 5, 9, 10, 12], dtype=np.int32)
    Ai = np.array([0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4], dtype=np.int32)
    Ax = np.array([2.0, 3.0, 3.0, -1.0, 4.0, 4.0, -3.0, 1.0, 2.0, 2.0, 6.0, 1.0])
    b = np.array([8.0, 45.0, -3.0, 3.0, 19.0])

    # Expected solution (computed from Fortran test)
    x_expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    return {"Ap": Ap, "Ai": Ai, "Ax": Ax, "b": b, "x": x_expected, "n": n}


@pytest.fixture
def dense_matrix(test_system):
    """Convert CSC to dense for verification."""
    d = test_system
    n = d["n"]
    A = np.zeros((n, n))
    for j in range(n):
        for p in range(d["Ap"][j], d["Ap"][j + 1]):
            A[d["Ai"][p], j] = d["Ax"][p]
    return A


class TestBLQMRSolve:
    """Tests for single RHS solver."""

    def test_basic_solve(self, test_system, dense_matrix):
        """Test basic solve returns correct solution."""
        d = test_system
        result = blqmr_solve(d["Ap"], d["Ai"], d["Ax"], d["b"], tol=1e-10)

        assert isinstance(result, BLQMRResult)
        assert result.converged
        assert result.flag == 0

        # Check solution accuracy
        residual = np.linalg.norm(dense_matrix @ result.x - d["b"])
        assert residual < 1e-8

    def test_convergence_info(self, test_system):
        """Test convergence information is returned."""
        d = test_system
        result = blqmr_solve(d["Ap"], d["Ai"], d["Ax"], d["b"])

        assert result.iter > 0
        assert result.relres < 1.0
        assert result.relres >= 0.0

    def test_tolerance(self, test_system, dense_matrix):
        """Test different tolerance values."""
        d = test_system

        for tol in [1e-4, 1e-8, 1e-12]:
            result = blqmr_solve(d["Ap"], d["Ai"], d["Ax"], d["b"], tol=tol)
            if result.converged:
                assert result.relres <= tol * 10  # Allow some margin

    def test_no_preconditioner(self, test_system, dense_matrix):
        """Test solving without preconditioner."""
        d = test_system
        result = blqmr_solve(
            d["Ap"], d["Ai"], d["Ax"], d["b"], use_precond=False, tol=1e-8, maxiter=100
        )

        # Should still converge (maybe slower)
        residual = np.linalg.norm(dense_matrix @ result.x - d["b"])
        assert residual < 1e-4

    def test_maxiter(self, test_system):
        """Test maximum iterations limit."""
        d = test_system
        result = blqmr_solve(d["Ap"], d["Ai"], d["Ax"], d["b"], maxiter=2, tol=1e-15)

        assert result.iter <= 2

    def test_fortran_indexing(self, test_system, dense_matrix):
        """Test with 1-based (Fortran) indexing."""
        d = test_system
        Ap_f = d["Ap"] + 1
        Ai_f = d["Ai"] + 1

        result = blqmr_solve(Ap_f, Ai_f, d["Ax"], d["b"], zero_based=False, tol=1e-10)

        residual = np.linalg.norm(dense_matrix @ result.x - d["b"])
        assert residual < 1e-8


class TestBLQMRSolveMulti:
    """Tests for multiple RHS solver."""

    def test_multi_rhs(self, test_system, dense_matrix):
        """Test solving with multiple right-hand sides."""
        d = test_system
        b1 = d["b"]
        b2 = np.array([18.0, 45.0, -3.0, 3.0, 19.0])
        B = np.column_stack([b1, b2])

        result = blqmr_solve_multi(d["Ap"], d["Ai"], d["Ax"], B, tol=1e-10)

        assert result.x.shape == (5, 2)

        # Check both solutions
        for i in range(2):
            residual = np.linalg.norm(dense_matrix @ result.x[:, i] - B[:, i])
            assert residual < 1e-8


class TestSciPyInterface:
    """Tests for SciPy-compatible interface."""

    def test_scipy_interface(self, test_system, dense_matrix):
        """Test SciPy-style interface."""
        pytest.importorskip("scipy")
        from scipy.sparse import csc_matrix
        from blocksolver import blqmr_scipy

        d = test_system
        A = csc_matrix(dense_matrix)

        x, flag = blqmr_scipy(A, d["b"], tol=1e-10)

        assert flag == 0
        residual = np.linalg.norm(dense_matrix @ x - d["b"])
        assert residual < 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
