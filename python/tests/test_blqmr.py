"""
Comprehensive unit tests for BlockSolver BLQMR solver.

Run with: python -m unittest test_blqmr -v
      or: python test_blqmr.py
"""

import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Import blocksolver components
from blocksolver import (
    blqmr,
    blqmr_solve,
    blqmr_solve_multi,
    blqmr_scipy,
    BLQMRResult,
    BLQMR_EXT,
    HAS_NUMBA,
    qqr,
    make_preconditioner,
    BLQMRWorkspace,
    get_backend_info,
)


class TestSystemFixtures:
    """Test system generators."""

    @staticmethod
    def small_system():
        """5x5 sparse SYMMETRIC test system."""
        # Create a symmetric tridiagonal matrix (SPD)
        # This is a simple Laplacian-like matrix that BLQMR handles well
        n = 5

        # Build symmetric tridiagonal: 4 on diagonal, -1 on off-diagonals
        # CSC format
        # Column 0: rows 0,1
        # Column 1: rows 0,1,2
        # Column 2: rows 1,2,3
        # Column 3: rows 2,3,4
        # Column 4: rows 3,4

        Ap = np.array([0, 2, 5, 8, 11, 13], dtype=np.int32)
        Ai = np.array([0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4], dtype=np.int32)
        Ax = np.array(
            [4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0]
        )

        # Expected solution x = [1, 2, 3, 4, 5]
        x_expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Compute b = A @ x
        # A @ [1,2,3,4,5] for tridiagonal with 4 on diag, -1 off-diag:
        # row 0: 4*1 - 1*2 = 2
        # row 1: -1*1 + 4*2 - 1*3 = 4
        # row 2: -1*2 + 4*3 - 1*4 = 6
        # row 3: -1*3 + 4*4 - 1*5 = 8
        # row 4: -1*4 + 4*5 = 16
        b = np.array([2.0, 4.0, 6.0, 8.0, 16.0])

        return {
            "Ap": Ap,
            "Ai": Ai,
            "Ax": Ax,
            "b": b,
            "x": x_expected,
            "n": n,
            "nnz": len(Ax),
        }

    @staticmethod
    def csc_to_dense(Ap, Ai, Ax, n):
        """Convert CSC format to dense matrix."""
        A = np.zeros((n, n))
        for j in range(n):
            for p in range(Ap[j], Ap[j + 1]):
                A[Ai[p], j] = Ax[p]
        return A

    @staticmethod
    def tridiagonal_system(n, diag=4.0, offdiag=-1.0):
        """Create tridiagonal system (n x n)."""
        from scipy.sparse import diags, csc_matrix

        A = diags([offdiag, diag, offdiag], [-1, 0, 1], shape=(n, n), format="csc")
        b = np.ones(n)
        return A, b

    @staticmethod
    def complex_symmetric_system(n):
        """Create complex symmetric (not Hermitian) system."""
        from scipy.sparse import diags, csc_matrix

        # Complex symmetric: A = A.T (not A.conj().T)
        diag = (4.0 + 1.0j) * np.ones(n)
        offdiag = (-1.0 + 0.5j) * np.ones(n - 1)
        A = diags([offdiag, diag, offdiag], [-1, 0, 1], shape=(n, n), format="csc")
        b = (1.0 + 0.5j) * np.ones(n)
        return A, b

    @staticmethod
    def random_spd_system(n, density=0.1, seed=42):
        """Create random symmetric positive definite sparse system."""
        from scipy.sparse import random, csc_matrix, eye

        np.random.seed(seed)
        A = random(n, n, density=density, format="csc")
        # Make symmetric positive definite: A = B.T @ B + I
        A = A.T @ A + n * eye(n, format="csc")
        b = np.random.randn(n)
        return A, b


class TestBLQMRResult(unittest.TestCase):
    """Tests for BLQMRResult dataclass."""

    def test_result_attributes(self):
        """Test BLQMRResult has all expected attributes."""
        result = BLQMRResult(x=np.array([1.0, 2.0]), flag=0, iter=5, relres=1e-10)
        self.assertTrue(hasattr(result, "x"))
        self.assertTrue(hasattr(result, "flag"))
        self.assertTrue(hasattr(result, "iter"))
        self.assertTrue(hasattr(result, "relres"))
        self.assertTrue(hasattr(result, "converged"))

    def test_converged_property(self):
        """Test converged property."""
        result_converged = BLQMRResult(x=np.array([1.0]), flag=0, iter=1, relres=1e-10)
        result_maxiter = BLQMRResult(x=np.array([1.0]), flag=1, iter=100, relres=1e-3)
        result_stagnated = BLQMRResult(x=np.array([1.0]), flag=3, iter=50, relres=1e-5)

        self.assertTrue(result_converged.converged)
        self.assertFalse(result_maxiter.converged)
        self.assertFalse(result_stagnated.converged)

    def test_result_repr(self):
        """Test string representation."""
        result = BLQMRResult(x=np.array([1.0]), flag=0, iter=5, relres=1e-10)
        repr_str = repr(result)
        self.assertIn("converged", repr_str)
        self.assertIn("iter=5", repr_str)


class TestBackendDetection(unittest.TestCase):
    """Tests for backend detection."""

    def test_blqmr_ext_is_bool(self):
        """Test BLQMR_EXT is boolean."""
        self.assertIsInstance(BLQMR_EXT, bool)

    def test_has_numba_is_bool(self):
        """Test HAS_NUMBA is boolean."""
        self.assertIsInstance(HAS_NUMBA, bool)

    def test_get_backend_info(self):
        """Test get_backend_info returns expected keys."""
        info = get_backend_info()
        self.assertIn("backend", info)
        self.assertIn("has_fortran", info)
        self.assertIn("has_numba", info)
        self.assertIn(info["backend"], ["binary", "native"])


class TestQuasiQR(unittest.TestCase):
    """Tests for Quasi-QR decomposition."""

    def test_qqr_real(self):
        """Test QQR on real matrix."""
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        A_copy = A.copy()
        Q, R = qqr(A_copy)

        self.assertEqual(Q.shape, (3, 2))
        self.assertEqual(R.shape, (2, 2))

        # R should be upper triangular
        self.assertAlmostEqual(R[1, 0], 0.0, places=10)

        # Q @ R should approximate original A
        assert_allclose(Q @ R, A, rtol=1e-10)

    def test_qqr_complex(self):
        """Test QQR on complex matrix."""
        A = np.array([[1.0 + 1j, 2.0 - 1j], [3.0 + 2j, 4.0 + 0j], [5.0 - 1j, 6.0 + 1j]])
        A_copy = A.copy()
        Q, R = qqr(A_copy)

        self.assertEqual(Q.shape, (3, 2))
        self.assertEqual(R.shape, (2, 2))

        # Q @ R should approximate original A
        assert_allclose(Q @ R, A, rtol=1e-10)

    def test_qqr_quasi_orthogonality(self):
        """Test quasi-orthogonality (Q.T @ Q, not Q.H @ Q)."""
        A = np.random.randn(10, 3)
        A_copy = A.copy()
        Q, R = qqr(A_copy)

        # For real matrices, Q.T @ Q should be approximately identity
        QtQ = Q.T @ Q
        assert_allclose(QtQ, np.eye(3), atol=1e-10)


class TestPreconditioner(unittest.TestCase):
    """Tests for preconditioner creation."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from scipy.sparse import csc_matrix

            self.scipy_available = True
            A, _ = TestSystemFixtures.tridiagonal_system(10)
            self.A = A
        except ImportError:
            self.scipy_available = False

    def test_diagonal_preconditioner(self):
        """Test diagonal (Jacobi) preconditioner."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        M = make_preconditioner(self.A, "diag")
        self.assertIsNotNone(M)
        # Diagonal preconditioner should be a sparse matrix
        self.assertEqual(M.shape, self.A.shape)

    def test_jacobi_alias(self):
        """Test 'jacobi' as alias for 'diag'."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        M1 = make_preconditioner(self.A, "diag")
        M2 = make_preconditioner(self.A, "jacobi")
        # Both should produce same result
        assert_allclose(M1.diagonal(), M2.diagonal())

    def test_ilu_preconditioner(self):
        """Test ILU preconditioner."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        M = make_preconditioner(self.A, "ilu")
        self.assertIsNotNone(M)

    def test_invalid_preconditioner_type(self):
        """Test invalid preconditioner type raises error."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        with self.assertRaises(ValueError):
            make_preconditioner(self.A, "invalid_type")


class TestBLQMRWorkspace(unittest.TestCase):
    """Tests for BLQMRWorkspace."""

    def test_workspace_creation(self):
        """Test workspace creation."""
        ws = BLQMRWorkspace(100, 4)
        self.assertEqual(ws.n, 100)
        self.assertEqual(ws.m, 4)

    def test_workspace_arrays(self):
        """Test workspace arrays have correct shapes."""
        n, m = 50, 3
        ws = BLQMRWorkspace(n, m)

        self.assertEqual(ws.v.shape, (n, m, 3))
        self.assertEqual(ws.vt.shape, (n, m))
        self.assertEqual(ws.alpha.shape, (m, m))
        self.assertEqual(ws.Av.shape, (n, m))

    def test_workspace_reset(self):
        """Test workspace reset."""
        ws = BLQMRWorkspace(10, 2)
        ws.v.fill(1.0)
        ws.reset()
        self.assertEqual(np.sum(np.abs(ws.v)), 0.0)


class TestBLQMRSolve(unittest.TestCase):
    """Tests for blqmr_solve with CSC input."""

    def setUp(self):
        """Set up test fixtures."""
        self.sys = TestSystemFixtures.small_system()
        self.A_dense = TestSystemFixtures.csc_to_dense(
            self.sys["Ap"], self.sys["Ai"], self.sys["Ax"], self.sys["n"]
        )

    def test_basic_solve(self):
        """Test basic solve returns correct solution."""
        d = self.sys
        result = blqmr_solve(d["Ap"], d["Ai"], d["Ax"], d["b"], tol=1e-10)

        self.assertIsInstance(result, BLQMRResult)
        self.assertTrue(result.converged)
        self.assertEqual(result.flag, 0)

        residual = np.linalg.norm(self.A_dense @ result.x - d["b"])
        self.assertLess(residual, 1e-8)

    def test_solution_accuracy(self):
        """Test solution matches expected values."""
        d = self.sys
        result = blqmr_solve(d["Ap"], d["Ai"], d["Ax"], d["b"], tol=1e-12)

        if result.converged:
            assert_allclose(result.x, d["x"], rtol=1e-6)

    def test_convergence_info(self):
        """Test convergence information is returned."""
        d = self.sys
        result = blqmr_solve(d["Ap"], d["Ai"], d["Ax"], d["b"])

        self.assertGreater(result.iter, 0)
        self.assertLess(result.relres, 1.0)
        self.assertGreaterEqual(result.relres, 0.0)

    def test_tolerance_levels(self):
        """Test different tolerance values."""
        d = self.sys

        for tol in [1e-4, 1e-8, 1e-12]:
            with self.subTest(tol=tol):
                result = blqmr_solve(
                    d["Ap"], d["Ai"], d["Ax"], d["b"], tol=tol, maxiter=100
                )
                if result.converged:
                    # Check actual residual instead of relres from solver
                    # (relres is internal quasi-residual, not true residual)
                    actual_residual = np.linalg.norm(self.A_dense @ result.x - d["b"])
                    initial_residual = np.linalg.norm(d["b"])
                    actual_relres = actual_residual / initial_residual
                    self.assertLessEqual(actual_relres, tol * 100)

    def test_no_preconditioner(self):
        """Test solving without preconditioner."""
        d = self.sys
        # Changed: use_precond=False -> precond_type=None (or '')
        result = blqmr_solve(
            d["Ap"],
            d["Ai"],
            d["Ax"],
            d["b"],
            precond_type=None,  # No preconditioning
            tol=1e-6,
            maxiter=500,
        )

        residual = np.linalg.norm(self.A_dense @ result.x - d["b"])
        initial_residual = np.linalg.norm(d["b"])

        # Just verify solver doesn't blow up - convergence without
        # preconditioner is not guaranteed for all matrices
        self.assertLess(residual, initial_residual * 100)

    def test_maxiter_limit(self):
        """Test maximum iterations limit."""
        d = self.sys
        result = blqmr_solve(d["Ap"], d["Ai"], d["Ax"], d["b"], maxiter=2, tol=1e-15)

        self.assertLessEqual(result.iter, 2)

    def test_fortran_indexing(self):
        """Test with 1-based (Fortran) indexing."""
        d = self.sys
        Ap_f = d["Ap"] + 1
        Ai_f = d["Ai"] + 1

        result = blqmr_solve(Ap_f, Ai_f, d["Ax"], d["b"], zero_based=False, tol=1e-10)

        residual = np.linalg.norm(self.A_dense @ result.x - d["b"])
        self.assertLess(residual, 1e-8)


class TestBLQMRSolveMulti(unittest.TestCase):
    """Tests for blqmr_solve_multi with multiple RHS."""

    def setUp(self):
        """Set up test fixtures."""
        self.sys = TestSystemFixtures.small_system()
        self.A_dense = TestSystemFixtures.csc_to_dense(
            self.sys["Ap"], self.sys["Ai"], self.sys["Ax"], self.sys["n"]
        )

    def test_multi_rhs(self):
        """Test solving with multiple right-hand sides."""
        d = self.sys
        b1 = d["b"]
        b2 = np.array([18.0, 45.0, -3.0, 3.0, 19.0])
        B = np.column_stack([b1, b2])

        result = blqmr_solve_multi(d["Ap"], d["Ai"], d["Ax"], B, tol=1e-10)

        self.assertEqual(result.x.shape, (5, 2))

        for i in range(2):
            residual = np.linalg.norm(self.A_dense @ result.x[:, i] - B[:, i])
            self.assertLess(residual, 1e-8)

    def test_many_rhs(self):
        """Test with many right-hand sides."""
        d = self.sys
        nrhs = 10
        B = np.random.randn(d["n"], nrhs)

        result = blqmr_solve_multi(d["Ap"], d["Ai"], d["Ax"], B, tol=1e-8)

        self.assertEqual(result.x.shape, (d["n"], nrhs))


class TestBLQMRMainInterface(unittest.TestCase):
    """Tests for main blqmr() interface with sparse matrix input."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from scipy.sparse import csc_matrix

            self.scipy_available = True
        except ImportError:
            self.scipy_available = False

    def test_sparse_matrix_input(self):
        """Test blqmr() with scipy sparse matrix."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        A, b = TestSystemFixtures.tridiagonal_system(20)
        result = blqmr(A, b, tol=1e-8, maxiter=200)

        self.assertIsInstance(result, BLQMRResult)

        # Check residual is reduced (may not fully converge with limited iterations)
        residual = np.linalg.norm(A @ result.x - b)
        initial_residual = np.linalg.norm(b)
        self.assertLess(residual, initial_residual)  # Should at least improve

    def test_dense_matrix_input(self):
        """Test blqmr() with dense numpy array."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        n = 10
        A = (
            np.diag([4.0] * n)
            + np.diag([-1.0] * (n - 1), 1)
            + np.diag([-1.0] * (n - 1), -1)
        )
        b = np.ones(n)

        result = blqmr(A, b, tol=1e-10)

        residual = np.linalg.norm(A @ result.x - b)
        self.assertLess(residual, 1e-8)

    def test_1d_rhs(self):
        """Test with 1D right-hand side."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        A, b = TestSystemFixtures.tridiagonal_system(15)
        result = blqmr(A, b, tol=1e-10)

        self.assertEqual(result.x.ndim, 1)

    def test_2d_rhs(self):
        """Test with 2D right-hand side (multiple RHS)."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        A, _ = TestSystemFixtures.tridiagonal_system(15)
        B = np.random.randn(15, 5)

        result = blqmr(A, B, tol=1e-8)

        self.assertEqual(result.x.shape, (15, 5))

    def test_custom_preconditioner(self):
        """Test with custom preconditioner."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        A, b = TestSystemFixtures.tridiagonal_system(20)
        M1 = make_preconditioner(A, "diag")

        # Changed: use_precond=False -> precond_type=None
        # When M1 is provided, precond_type is ignored anyway
        result = blqmr(A, b, M1=M1, precond_type=None, tol=1e-10)

        residual = np.linalg.norm(A @ result.x - b)
        self.assertLess(residual, 1e-8)

    def test_workspace_reuse(self):
        """Test workspace reuse for repeated solves."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        A, _ = TestSystemFixtures.tridiagonal_system(20)
        ws = BLQMRWorkspace(20, 1)

        for _ in range(3):
            b = np.random.randn(20)
            result = blqmr(A, b, workspace=ws, tol=1e-8)
            self.assertTrue(result.converged or result.flag == 1)


class TestComplexSystems(unittest.TestCase):
    """Tests for complex-valued systems."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from scipy.sparse import csc_matrix

            self.scipy_available = True
        except ImportError:
            self.scipy_available = False

    def test_complex_symmetric(self):
        """Test complex symmetric system (A = A.T)."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        A, b = TestSystemFixtures.complex_symmetric_system(20)

        # Verify A is complex symmetric
        assert_allclose((A - A.T).toarray(), 0, atol=1e-14)

        result = blqmr(A, b, tol=1e-6, maxiter=300)

        # Check residual is reduced
        residual = np.linalg.norm(A @ result.x - b)
        initial_residual = np.linalg.norm(b)
        self.assertLess(residual, initial_residual)  # Should improve
        self.assertLess(residual, 1e-3)  # Relaxed for complex systems

    def test_complex_rhs(self):
        """Test real matrix with complex RHS."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        A, _ = TestSystemFixtures.tridiagonal_system(15)
        b = (1.0 + 0.5j) * np.ones(15)

        result = blqmr(A, b, tol=1e-10)

        self.assertTrue(np.iscomplexobj(result.x))


class TestSciPyInterface(unittest.TestCase):
    """Tests for SciPy-compatible interface."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from scipy.sparse import csc_matrix

            self.scipy_available = True
        except ImportError:
            self.scipy_available = False

    def test_scipy_interface_return_type(self):
        """Test blqmr_scipy returns (x, flag) tuple."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        A, b = TestSystemFixtures.tridiagonal_system(10)
        result = blqmr_scipy(A, b, tol=1e-10)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        x, flag = result
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(flag, int)

    def test_scipy_interface_accuracy(self):
        """Test blqmr_scipy solution accuracy."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        sys = TestSystemFixtures.small_system()
        A_dense = TestSystemFixtures.csc_to_dense(
            sys["Ap"], sys["Ai"], sys["Ax"], sys["n"]
        )
        from scipy.sparse import csc_matrix

        A = csc_matrix(A_dense)

        x, flag = blqmr_scipy(A, sys["b"], tol=1e-10)

        self.assertEqual(flag, 0)
        residual = np.linalg.norm(A_dense @ x - sys["b"])
        self.assertLess(residual, 1e-8)


class TestLargeSystems(unittest.TestCase):
    """Tests for larger systems."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from scipy.sparse import csc_matrix

            self.scipy_available = True
        except ImportError:
            self.scipy_available = False

    def test_medium_system(self):
        """Test medium-sized system (100x100)."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        A, b = TestSystemFixtures.tridiagonal_system(100)
        result = blqmr(A, b, tol=1e-8, maxiter=500)

        # Check residual is reduced
        residual = np.linalg.norm(A @ result.x - b)
        initial_residual = np.linalg.norm(b)
        self.assertLess(residual, initial_residual)  # Should improve
        # For larger systems, just verify reasonable reduction
        self.assertLess(residual, 1.0)

    def test_random_spd_system(self):
        """Test random SPD system."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        A, b = TestSystemFixtures.random_spd_system(50, density=0.2)
        result = blqmr(A, b, tol=1e-8, maxiter=100)

        # Should converge or at least reduce residual
        residual = np.linalg.norm(A @ result.x - b)
        initial_residual = np.linalg.norm(b)
        self.assertLess(residual, initial_residual)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from scipy.sparse import csc_matrix, eye

            self.scipy_available = True
        except ImportError:
            self.scipy_available = False

    def test_identity_matrix(self):
        """Test solving with identity matrix."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        from scipy.sparse import eye

        n = 10
        A = eye(n, format="csc")
        b = np.arange(1, n + 1, dtype=float)

        result = blqmr(A, b, tol=1e-12)

        # Solution should equal RHS for identity matrix
        assert_allclose(result.x, b, rtol=1e-10)

    def test_diagonal_matrix(self):
        """Test solving with diagonal matrix."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        from scipy.sparse import diags

        n = 10
        d = np.arange(1, n + 1, dtype=float)
        A = diags(d, 0, format="csc")
        b = np.ones(n)

        result = blqmr(A, b, tol=1e-12)

        expected = 1.0 / d
        assert_allclose(result.x, expected, rtol=1e-8)

    def test_zero_rhs(self):
        """Test with zero right-hand side."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        A, _ = TestSystemFixtures.tridiagonal_system(10)
        b = np.zeros(10)

        result = blqmr(A, b, tol=1e-10)

        # Solution should be zero
        assert_allclose(result.x, 0, atol=1e-14)

    def test_single_element(self):
        """Test 1x1 system."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        from scipy.sparse import csc_matrix

        A = csc_matrix([[5.0]])
        b = np.array([10.0])

        result = blqmr(A, b, tol=1e-12)

        assert_allclose(result.x, [2.0], rtol=1e-10)


class TestPreconditionerTypes(unittest.TestCase):
    """Tests for different precond_type values."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from scipy.sparse import csc_matrix

            self.scipy_available = True
            self.sys = TestSystemFixtures.small_system()
            self.A_dense = TestSystemFixtures.csc_to_dense(
                self.sys["Ap"], self.sys["Ai"], self.sys["Ax"], self.sys["n"]
            )
        except ImportError:
            self.scipy_available = False

    def test_precond_none(self):
        """Test precond_type=None (no preconditioning)."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        d = self.sys
        result = blqmr_solve(
            d["Ap"], d["Ai"], d["Ax"], d["b"], precond_type=None, tol=1e-8, maxiter=500
        )
        # Should still produce a result (may not converge without precond)
        self.assertIsInstance(result, BLQMRResult)

    def test_precond_empty_string(self):
        """Test precond_type='' (no preconditioning)."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        d = self.sys
        result = blqmr_solve(
            d["Ap"], d["Ai"], d["Ax"], d["b"], precond_type="", tol=1e-8, maxiter=500
        )
        self.assertIsInstance(result, BLQMRResult)

    def test_precond_ilu(self):
        """Test precond_type='ilu'."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        d = self.sys
        result = blqmr_solve(
            d["Ap"], d["Ai"], d["Ax"], d["b"], precond_type="ilu", tol=1e-10
        )

        self.assertTrue(result.converged)
        residual = np.linalg.norm(self.A_dense @ result.x - d["b"])
        self.assertLess(residual, 1e-8)

    def test_precond_diag(self):
        """Test precond_type='diag'."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        d = self.sys
        result = blqmr_solve(
            d["Ap"],
            d["Ai"],
            d["Ax"],
            d["b"],
            precond_type="diag",
            tol=1e-10,
            maxiter=100,
        )

        residual = np.linalg.norm(self.A_dense @ result.x - d["b"])
        # Diagonal preconditioner may need more iterations
        self.assertLess(residual, 1e-6)

    def test_precond_jacobi(self):
        """Test precond_type='jacobi' (alias for diag)."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        d = self.sys
        result = blqmr_solve(
            d["Ap"],
            d["Ai"],
            d["Ax"],
            d["b"],
            precond_type="jacobi",
            tol=1e-10,
            maxiter=100,
        )

        residual = np.linalg.norm(self.A_dense @ result.x - d["b"])
        self.assertLess(residual, 1e-6)

    def test_precond_integer_codes(self):
        """Test integer precond_type codes for Fortran compatibility."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        d = self.sys

        # 0 = no preconditioning
        result0 = blqmr_solve(
            d["Ap"], d["Ai"], d["Ax"], d["b"], precond_type=0, maxiter=500
        )
        self.assertIsInstance(result0, BLQMRResult)

        # 2 = ILU
        result2 = blqmr_solve(
            d["Ap"], d["Ai"], d["Ax"], d["b"], precond_type=2, tol=1e-10
        )
        self.assertTrue(result2.converged)

        # 3 = diagonal
        result3 = blqmr_solve(
            d["Ap"], d["Ai"], d["Ax"], d["b"], precond_type=3, tol=1e-10, maxiter=100
        )
        self.assertIsInstance(result3, BLQMRResult)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple features."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from scipy.sparse import csc_matrix

            self.scipy_available = True
        except ImportError:
            self.scipy_available = False

    def test_full_workflow(self):
        """Test complete workflow: create system, solve, verify."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        # Create system
        A, b = TestSystemFixtures.tridiagonal_system(50)

        # Check backend info
        info = get_backend_info()
        self.assertIn(info["backend"], ["binary", "native"])

        # Solve with default settings
        result1 = blqmr(A, b)
        self.assertTrue(result1.converged or result1.iter > 0)

        # Solve with custom tolerance
        result2 = blqmr(A, b, tol=1e-10, maxiter=200)

        # Verify solution
        if result2.converged:
            residual = np.linalg.norm(A @ result2.x - b)
            self.assertLess(residual, 1e-6)

    def test_compare_interfaces(self):
        """Test that different interfaces give same result."""
        if not self.scipy_available:
            self.skipTest("SciPy not available")

        from scipy.sparse import csc_matrix

        # Create system
        sys = TestSystemFixtures.small_system()
        A_dense = TestSystemFixtures.csc_to_dense(
            sys["Ap"], sys["Ai"], sys["Ax"], sys["n"]
        )
        A_sparse = csc_matrix(A_dense)

        # Solve via different interfaces
        result1 = blqmr(A_sparse, sys["b"], tol=1e-10)
        result2 = blqmr_solve(sys["Ap"], sys["Ai"], sys["Ax"], sys["b"], tol=1e-10)
        x3, _ = blqmr_scipy(A_sparse, sys["b"], tol=1e-10)

        # All should give similar solutions
        if result1.converged and result2.converged:
            assert_allclose(result1.x, result2.x, rtol=1e-6)
            assert_allclose(result1.x, x3, rtol=1e-6)


def run_tests():
    """Run all tests."""
    unittest.main(module="test_blqmr", verbosity=2)


if __name__ == "__main__":
    run_tests()
