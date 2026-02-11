![](https://neurojson.org/wiki/upload/neurojson_banner_long.png)

# BLIT - Block Iterative Linear Solvers

**BLIT** (Block Iterative Linear Solvers) is a high-performance library for solving large sparse linear systems using the Block Quasi-Minimal Residual (BL-QMR) algorithm. It provides unified interfaces across Fortran, Python, MATLAB, and C++.

- Copyright: (C) Qianqian Fang (2005, 2011, 2026) <q.fang at neu.edu>
- License: BSD-3-Clause and GPL-v3 dual-licensed
- Version: 0.6.0
- Website: https://neurojson.org/Page/blocksolver
- Github: https://github.com/fangq/blit
- Acknowledgement: This project is supported by US National Institute of Health (NIH)
  grant [U24-NS124027](https://reporter.nih.gov/project-details/10308329)

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [When to Use BLIT](#when-to-use-blit)
- [Installation](#installation)
- [Python Interface](#python-interface)
- [MATLAB Interface](#matlab-interface)
- [Fortran Interface](#fortran-interface)
- [C++ Interface](#c-interface)
- [Preconditioning Guide](#preconditioning-guide)
- [Benchmarks](#benchmarks)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Algorithm Details](#algorithm-details)
- [Citation](#citation)
- [License](#license)

---

## Introduction

### The Problem BLIT Solves

Many scientific and engineering applications require solving sparse linear systems of the form **Ax = b**, where:

- **A** is a large, sparse matrix (thousands to millions of unknowns)
- **A** is **symmetric** (A = Aᵀ) but potentially **complex** and **indefinite**
- Multiple right-hand sides **b₁, b₂, ..., bₘ** need to be solved simultaneously

Traditional iterative solvers like Conjugate Gradient (CG) require **symmetric positive definite** matrices, while GMRES works for general matrices but doesn't exploit symmetry. For **complex symmetric** matrices—common in frequency-domain finite element methods—neither is optimal.

### Why Block QMR?

The **Block Quasi-Minimal Residual (BL-QMR)** algorithm addresses this gap:

1. **Exploits Complex Symmetry**: Uses the quasi inner product ⟨x,y⟩ = Σ xₖyₖ (without conjugation), which is the natural inner product for complex symmetric systems.

2. **Block Acceleration**: Solves multiple right-hand sides simultaneously, sharing Krylov subspace information. With m RHS vectors, BLQMR typically requires only 20-30% of the iterations that m separate solves would need.

3. **Smooth Convergence**: The quasi-minimal residual property provides monotonically decreasing residuals without the erratic behavior of some Krylov methods.

4. **Memory Efficient**: Uses short three-term recurrences rather than storing the full Krylov basis like GMRES.

### Target Applications

BLIT excels in these domains:

| Application | Why BLIT? |
|-------------|-----------|
| **Diffuse Optical Tomography (DOT)** | Complex symmetric diffusion matrices, many source positions |
| **Frequency-Domain Electromagnetics** | Helmholtz equation produces complex symmetric systems |
| **Acoustic Scattering** | Wave equation discretization with absorbing boundaries |
| **Microwave Imaging** | Original application—medical imaging with multiple antenna sources |
| **Structural Dynamics** | Frequency response analysis with many load cases |
| **Seismic Inversion** | Multiple shot gathers solved simultaneously |

### Library Overview

BLIT provides consistent interfaces across multiple programming languages:

| Language | Implementation | Backend | Key Features |
|----------|---------------|---------|--------------|
| **Python** | `blocksolver` package | Fortran (fast) or pure Python (portable) | NumPy/SciPy integration, automatic fallback |
| **MATLAB** | `blqmr.m` | Native MATLAB | Drop-in replacement for `qmr()`, sparse matrix support |
| **Fortran 90** | `blit_blqmr_*.f90` | Native | Maximum performance, UMFPACK integration |
| **C++** | `blit_solvers.h` | Fortran via templates | Type-safe wrappers, RAII memory management |

All implementations share the same algorithm and produce equivalent results within numerical precision.

---

## Key Features

- **Block Algorithm**: Solves multiple right-hand sides simultaneously with shared Krylov subspace
- **Complex Symmetric Support**: Designed for A = Aᵀ matrices (not Hermitian A = A†)
- **Built-in Preconditioning**: ILU, diagonal (Jacobi), and SSOR with automatic setup
- **Flexible Backends**: High-performance Fortran core with pure Python/MATLAB fallbacks
- **Optional Acceleration**: Numba JIT compilation for Python backend
- **Production Ready**: Used in published research since 2004

---

## When to Use BLIT

| Problem Type | Recommendation |
|--------------|----------------|
| Complex symmetric FEM matrices (A = Aᵀ) | ✅ **Ideal** - designed for this |
| Multiple right-hand sides (4-64+) | ✅ **Ideal** - block algorithm shines |
| Frequency-domain wave problems | ✅ **Excellent** - Helmholtz, acoustics, EM |
| Diffuse optical tomography | ✅ **Excellent** - original application |
| Real symmetric positive definite | Consider CG first, BLIT as alternative |
| General non-symmetric systems | Use GMRES or BiCGSTAB instead |
| Hermitian systems (A = A†) | Use MINRES or CG |

---

## Installation

### Python

```bash
# From PyPI (pure Python fallback, works everywhere)
pip install blocksolver

# Optional: Numba acceleration for Python backend
pip install numba
```

**Build with Fortran backend (recommended for production):**

```bash
# Ubuntu/Debian - install dependencies
sudo apt install gfortran libsuitesparse-dev libblas-dev liblapack-dev

# macOS - install dependencies
brew install gcc suite-sparse openblas

# Build and install
cd python && pip install .
```

### MATLAB

Add the `matlab/` directory to your MATLAB path:

```matlab
addpath('/path/to/blit/matlab');
```

### Fortran / C++

Build the Fortran library and link against it:

```bash
cd src && make
# Creates libblit.a

# Link with: -lblit -lumfpack -lblas -llapack -lgfortran
```

---

## Python Interface

### Installation Check

```python
from blocksolver import BLQMR_EXT, HAS_NUMBA, get_backend_info

# Check which backends are available
print(get_backend_info())
# Output: {'backend': 'binary', 'has_fortran': True, 'has_numba': True}

# BLQMR_EXT = True means Fortran backend is active (10-100x faster)
# HAS_NUMBA = True means Numba JIT is available for Python backend
```

### Basic Usage

```python
import numpy as np
from scipy.sparse import diags, csc_matrix
from blocksolver import blqmr

# Create a symmetric tridiagonal matrix (1000 x 1000)
n = 1000
A = diags(
    [-1, 4, -1],           # Diagonal values: sub, main, super
    [-1, 0, 1],            # Diagonal offsets
    shape=(n, n),
    format='csc'           # CSC format is optimal for sparse solvers
)

# Single right-hand side
b = np.random.randn(n)

# Solve Ax = b
result = blqmr(
    A,                     # Sparse or dense matrix (n x n)
    b,                     # Right-hand side vector (n,) or matrix (n x m)
    tol=1e-10,             # Convergence tolerance for relative residual
    maxiter=500,           # Maximum iterations (default: n)
    precond_type='ilu'     # Preconditioner: 'ilu', 'diag', or None
)

# Check results
print(f"Converged: {result.converged}")    # True if flag == 0
print(f"Iterations: {result.iter}")         # Number of iterations performed
print(f"Relative residual: {result.relres:.2e}")  # Final ||Ax-b||/||b||

# Verify solution accuracy
true_residual = np.linalg.norm(A @ result.x - b) / np.linalg.norm(b)
print(f"True residual: {true_residual:.2e}")
```

### Multiple Right-Hand Sides (Block Solve)

```python
import numpy as np
from scipy.sparse import random as sparse_random
from blocksolver import blqmr

# Create a complex symmetric matrix (common in frequency-domain FEM)
n = 5000
# Random symmetric: A = B + B.T to ensure symmetry
B = sparse_random(n, n, density=0.01, format='csc')
A = (B + B.T) / 2 + 1j * sparse_random(n, n, density=0.005, format='csc')
A = (A + A.T) / 2  # Ensure complex symmetric (A = A.T, not A.conj().T)

# 16 right-hand sides (e.g., 16 source positions in imaging)
num_sources = 16
B = np.random.randn(n, num_sources) + 1j * np.random.randn(n, num_sources)

# Block solve - all 16 RHS solved together, sharing Krylov information
result = blqmr(
    A,                     # Complex symmetric matrix
    B,                     # Multiple RHS as columns (n x 16)
    tol=1e-8,              # Tolerance applies to worst-case RHS
    maxiter=1000,          # Max block iterations
    precond_type='diag'    # Diagonal preconditioner (ILU may fail for complex)
)

print(f"Solution shape: {result.x.shape}")  # (5000, 16)
print(f"Block iterations: {result.iter}")   # Much less than 16 * single iterations
print(f"All converged: {result.converged}")

# Verify each solution
for i in range(num_sources):
    res = np.linalg.norm(A @ result.x[:, i] - B[:, i]) / np.linalg.norm(B[:, i])
    print(f"  RHS {i+1}: residual = {res:.2e}")
```

### Custom Preconditioning

```python
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spilu
from blocksolver import blqmr, make_preconditioner, SparsePreconditioner

# Create test matrix
n = 2000
A = diags([-1, 4, -1], [-1, 0, 1], shape=(n, n), format='csc')
b = np.random.randn(n)

# Option 1: Use built-in preconditioner factory
M_diag = make_preconditioner(A, 'diag')   # Diagonal (Jacobi)
M_ilu = make_preconditioner(A, 'ilu')     # Incomplete LU

result = blqmr(A, b, M1=M_diag, precond_type=None)  # precond_type=None when M1 provided

# Option 2: Split preconditioning for symmetric systems
# This preserves symmetry of the preconditioned system
d = np.abs(A.diagonal())
d[d < 1e-14] = 1.0
sqrt_d = np.sqrt(d)
M_sqrt = diags(sqrt_d, format='csc')

result = blqmr(
    A, b,
    M1=M_sqrt,             # Left preconditioner: M1^{-1} * A * M2^{-1}
    M2=M_sqrt,             # Right preconditioner: recovers x = M2^{-1} * y
    precond_type=None      # Disable auto-preconditioner when M1/M2 provided
)

# Option 3: Custom ILU with tuned parameters
ilu_factor = spilu(
    A.tocsc(),
    drop_tol=1e-4,         # Drop entries smaller than this (smaller = stronger)
    fill_factor=10         # Allow up to 10x fill-in
)
M_custom = SparsePreconditioner(ilu_factor)

result = blqmr(A, b, M1=M_custom, precond_type=None)
```

### Low-Level CSC Interface

```python
import numpy as np
from blocksolver import blqmr_solve, blqmr_solve_multi

# Direct CSC format input (useful when matrix is already in this form)
# CSC format: Ap[j] to Ap[j+1]-1 gives row indices for column j

# Example: 5x5 sparse matrix
n = 5
Ap = np.array([0, 2, 5, 9, 10, 12], dtype=np.int32)  # Column pointers (length n+1)
Ai = np.array([0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4], dtype=np.int32)  # Row indices
Ax = np.array([2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1.])    # Values
b = np.array([8., 45., -3., 3., 19.])

# Single RHS
result = blqmr_solve(
    Ap, Ai, Ax, b,
    tol=1e-8,              # Convergence tolerance
    maxiter=100,           # Maximum iterations
    droptol=0.001,         # ILU drop tolerance (Fortran backend only)
    precond_type='ilu',    # 'ilu', 'diag', or None
    zero_based=True        # True for 0-based indexing (Python/C convention)
)

# Multiple RHS
B = np.random.randn(n, 4)
result = blqmr_solve_multi(
    Ap, Ai, Ax, B,
    tol=1e-8,
    zero_based=True
)
```

### SciPy-Compatible Interface

```python
from blocksolver import blqmr_scipy

# Drop-in replacement for scipy.sparse.linalg solvers
# Returns (x, flag) tuple like scipy's qmr, gmres, etc.
x, flag = blqmr_scipy(A, b, tol=1e-10, maxiter=500)

if flag == 0:
    print("Converged!")
else:
    print(f"Failed with flag {flag}")
```

### API Reference

#### `blqmr(A, B, **kwargs) → BLQMRResult`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | sparse/ndarray | required | Symmetric n×n matrix (real or complex) |
| `B` | ndarray | required | RHS vector (n,) or matrix (n×m) |
| `tol` | float | 1e-6 | Convergence tolerance for relative residual |
| `maxiter` | int | n | Maximum iterations |
| `M1` | preconditioner | None | Left preconditioner (M1⁻¹ applied to residual) |
| `M2` | preconditioner | None | Right preconditioner (for split preconditioning) |
| `x0` | ndarray | zeros | Initial guess |
| `precond_type` | str/None | 'ilu' | Auto-preconditioner: 'ilu', 'diag', None |
| `droptol` | float | 0.001 | ILU drop tolerance (Fortran backend only) |
| `residual` | bool | False | Use true residual (slower but more accurate) |
| `workspace` | BLQMRWorkspace | None | Pre-allocated workspace for repeated solves |

**Returns:** `BLQMRResult` dataclass with:

| Attribute | Type | Description |
|-----------|------|-------------|
| `x` | ndarray | Solution array (n,) or (n, m) |
| `flag` | int | 0=converged, 1=maxiter, 2=precond fail, 3=stagnated |
| `iter` | int | Iterations performed |
| `relres` | float | Final relative residual |
| `converged` | bool | True if flag == 0 |
| `resv` | ndarray | Residual history (Python backend only) |

---

## MATLAB Interface

### Basic Usage

```matlab
% Create a symmetric tridiagonal matrix
n = 1000;
e = ones(n, 1);
A = spdiags([-e, 4*e, -e], -1:1, n, n);  % Tridiagonal: -1, 4, -1

% Single right-hand side
b = rand(n, 1);

% Basic solve with default parameters
x = blqmr(A, b);

% Solve with explicit parameters
[x, flag, relres, iter, resvec] = blqmr(...
    A, ...           % Sparse symmetric matrix (n x n)
    b, ...           % Right-hand side vector (n x 1) or matrix (n x m)
    1e-10, ...       % qtol: quasi-residual tolerance (default: 1e-6)
    500, ...         % maxit: maximum iterations (default: min(n, 20))
    [], ...          % M1: left preconditioner (optional)
    [], ...          % M2: right preconditioner (optional, M = M1*M2)
    [] ...           % x0: initial guess (default: zeros)
);

% Check convergence
if flag == 0
    fprintf('Converged in %d iterations, relres = %.2e\n', iter, relres);
else
    fprintf('Failed: flag=%d, iter=%d, relres=%.2e\n', flag, iter, relres);
end

% Plot convergence history
semilogy(resvec);
xlabel('Iteration'); ylabel('Quasi-residual');
title('BLQMR Convergence');
```

### Multiple Right-Hand Sides

```matlab
% Create complex symmetric matrix (frequency-domain FEM)
n = 5000;
% Real stiffness matrix
K = gallery('poisson', round(sqrt(n)));
K = K(1:n, 1:n);
% Add complex mass term (lossy medium)
omega = 2 * pi * 100e6;  % 100 MHz
M = speye(n) * 0.01;
A = K + 1i * omega * M;  % Complex symmetric: A = A.' (not A')

% 16 source positions (e.g., antenna array)
num_sources = 16;
B = randn(n, num_sources) + 1i * randn(n, num_sources);

% Block solve - all sources solved together
[X, flag, relres, iter] = blqmr(A, B, 1e-8, 1000);

fprintf('Solved %d systems in %d block iterations\n', num_sources, iter);
fprintf('Solution size: %d x %d\n', size(X, 1), size(X, 2));

% Verify each solution
for i = 1:num_sources
    res = norm(A * X(:,i) - B(:,i)) / norm(B(:,i));
    fprintf('  RHS %2d: residual = %.2e\n', i, res);
end
```

### Preconditioning Options

```matlab
% Create test matrix
n = 3000;
A = gallery('wathen', 30, 30);  % Sparse symmetric positive definite
A = A(1:n, 1:n);
b = rand(n, 1);

% Option 1: Automatic preconditioner via opt structure
opt.precond = 'ilu';       % 'ilu', 'ilutp', 'ichol', or 'diag'
opt.droptol = 1e-3;        % Drop tolerance for ILU/ILUTP
[x, flag] = blqmr(A, b, 1e-10, 500, [], [], [], opt);

% Option 2: Manual ILU preconditioner
[L, U] = ilu(A, struct('type', 'ilutp', 'droptol', 1e-4));
[x, flag] = blqmr(A, b, 1e-10, 500, L, U);

% Option 3: Split Jacobi for symmetric systems (preserves symmetry)
d = diag(A);
d(abs(d) < 1e-14) = 1;     % Handle near-zero diagonal
M = spdiags(sqrt(d), 0, n, n);  % M1 = M2 = sqrt(D)
[x, flag, relres, iter] = blqmr(A, b, 1e-10, 500, M, M);
fprintf('Split Jacobi: %d iterations, relres=%.2e\n', iter, relres);

% Option 4: Incomplete Cholesky for SPD matrices
try
    L = ichol(A, struct('type', 'ict', 'droptol', 1e-3));
    [x, flag] = blqmr(A, b, 1e-10, 500, L, L');
catch ME
    fprintf('ichol failed: %s\n', ME.message);
end
```

### Batch Processing for Robustness

```matlab
% For very ill-conditioned or complex symmetric systems,
% processing RHS in smaller batches can improve stability

n = 5000;
A = create_complex_fem_matrix(n);  % Your application
B = rand(n, 64);  % 64 right-hand sides

% Solve in batches of 8 (more robust than 64 at once)
opt.blocksize = 8;         % Process 8 RHS per batch
opt.precond = 'diag';      % Diagonal preconditioner
[X, flag, relres, iter] = blqmr(A, B, 1e-8, 1000, [], [], [], opt);

fprintf('Solved in %d iterations (flag=%d)\n', iter, flag);
```

### Options Structure Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `opt.precond` | string | none | Auto-create preconditioner: 'ilu', 'ilutp', 'ichol', 'diag' |
| `opt.droptol` | float | 1e-3 | Drop tolerance for ILU/ILUTP/ICHOL |
| `opt.residual` | 0/1 | 0 | Use true residual (1) vs quasi-residual (0) |
| `opt.blocksize` | int | all | Process RHS in batches of this size |
| `opt.michol` | string | - | Modified incomplete Cholesky option |

### Return Flags

| Flag | Meaning | Action |
|------|---------|--------|
| 0 | Converged below tolerance | Success |
| 1 | Maximum iterations reached | Increase `maxit` or strengthen preconditioner |
| 2 | Preconditioner is rank-deficient | Use different preconditioner |
| 3 | Stagnated (residual stopped decreasing) | Check matrix properties |

---

## Fortran Interface

### Direct Fortran Usage

```fortran
program test_blqmr
    use blit_precision          ! Kind parameters (Kdouble)
    use blit_blqmr_real         ! Real double precision solver
    implicit none

    ! Solver object and parameters
    type(BLQMRSolver) :: qmr
    integer :: n, nnz, nrhs, i

    ! CSC format arrays (1-based indexing)
    integer, allocatable :: Ap(:), Ai(:)
    real(kind=Kdouble), allocatable :: Ax(:), B(:,:), X(:,:)

    ! Problem setup: 5x5 sparse matrix
    n = 5
    nnz = 12
    nrhs = 2

    allocate(Ap(n+1), Ai(nnz), Ax(nnz))
    allocate(B(n, nrhs), X(n, nrhs))

    ! Column pointers (1-based for Fortran)
    Ap = [1, 3, 6, 10, 11, 13]

    ! Row indices
    Ai = [1, 2, 1, 3, 5, 2, 3, 4, 5, 3, 2, 5]

    ! Non-zero values
    Ax = [2.d0, 3.d0, 3.d0, -1.d0, 4.d0, 4.d0, -3.d0, 1.d0, 2.d0, 2.d0, 6.d0, 1.d0]

    ! Right-hand sides
    B(:,1) = [8.d0, 45.d0, -3.d0, 3.d0, 19.d0]
    B(:,2) = [1.d0, 2.d0, 3.d0, 4.d0, 5.d0]

    ! Initialize solution to zero
    X = 0.0d0

    ! Create solver object
    call BLQMRCreate(qmr, n)

    ! Set solver parameters
    qmr%maxit = 100          ! Maximum iterations
    qmr%qtol = 1.0d-10       ! Convergence tolerance
    qmr%droptol = 1.0d-3     ! ILU drop tolerance
    qmr%pcond_type = 2       ! Preconditioner: 0=none, 2=ILU, 3=diagonal
    qmr%isquasires = 0       ! 0=quasi-residual, 1=true residual

    ! Prepare preconditioner (factorize once, reuse for multiple solves)
    call BLQMRPrep(qmr, Ap, Ai, Ax, nnz)

    ! Solve the system
    call BLQMRSolve(qmr, Ap, Ai, Ax, nnz, X, B, nrhs)

    ! Check results
    print '(A,I0,A,I0,A,ES10.2)', 'Flag: ', qmr%flag, &
          ', Iterations: ', qmr%iter, ', Relres: ', qmr%relres

    ! Print solution
    do i = 1, nrhs
        print '(A,I0,A,5ES12.4)', 'Solution ', i, ': ', X(:,i)
    end do

    ! Cleanup
    call BLQMRDestroy(qmr)
    deallocate(Ap, Ai, Ax, B, X)

end program test_blqmr
```

### Complex Symmetric Systems

```fortran
program test_blqmr_complex
    use blit_precision
    use blit_blqmr_complex      ! Complex double precision solver
    implicit none

    type(BLQMRSolver) :: qmr
    integer :: n, nnz, nrhs

    integer, allocatable :: Ap(:), Ai(:)
    complex(kind=Kdouble), allocatable :: Ax(:), B(:,:), X(:,:)

    ! ... setup complex CSC arrays ...

    call BLQMRCreate(qmr, n)
    qmr%maxit = 500
    qmr%qtol = 1.0d-8
    qmr%pcond_type = 3        ! Diagonal preconditioner (safer for complex)

    call BLQMRPrep(qmr, Ap, Ai, Ax, nnz)
    call BLQMRSolve(qmr, Ap, Ai, Ax, nnz, X, B, nrhs)

    call BLQMRDestroy(qmr)

end program test_blqmr_complex
```

### F2PY Wrapper Subroutines

These subroutines provide the Python-Fortran interface:

```fortran
! Single RHS, real
subroutine blqmr_solve_real(n, nnz, Ap, Ai, Ax, b, x, &
                            maxit, qtol, droptol, pcond_type, &
                            flag, iter, relres)

! Multiple RHS, real
subroutine blqmr_solve_real_multi(n, nnz, nrhs, Ap, Ai, Ax, B, X, &
                                   maxit, qtol, droptol, pcond_type, &
                                   flag, iter, relres)

! Single RHS, complex
subroutine blqmr_solve_complex(n, nnz, Ap, Ai, Ax, b, x, &
                               maxit, qtol, droptol, pcond_type, &
                               flag, iter, relres)

! Multiple RHS, complex
subroutine blqmr_solve_complex_multi(n, nnz, nrhs, Ap, Ai, Ax, B, X, &
                                      maxit, qtol, droptol, pcond_type, &
                                      flag, iter, relres)
```

---

## C++ Interface

### Template-Based Wrapper

```cpp
#include "blit_solvers.h"
#include <vector>
#include <iostream>

int main() {
    // Problem dimensions
    const int n = 1000;      // Matrix size
    const int nnz = 2998;    // Non-zeros
    const int nrhs = 4;      // Right-hand sides

    // CSC format arrays (1-based indexing for Fortran backend)
    std::vector<int> Ap(n + 1);
    std::vector<int> Ai(nnz);
    std::vector<double> Ax(nnz);
    std::vector<double> b(n * nrhs);
    std::vector<double> x(n * nrhs);

    // ... fill matrix data (1-based indexing) ...

    // Create solver for real double precision
    // Template parameter: double for real, F90Complex for complex
    BlitBLQMR<double> solver(n, nrhs,
        100,      // maxit: maximum iterations
        1e-3,     // droptol: ILU drop tolerance
        0,        // isquasires: 0=quasi, 1=true residual
        1         // debug: print convergence info
    );

    // Prepare: factorize preconditioner
    // Arrays are copied internally, originals can be freed
    solver.Prepare(Ap.data(), Ai.data(), Ax.data(), nnz);

    // Solve: x = A^{-1} * b
    solver.Solve(x.data(), b.data(), nrhs);

    // Print convergence info
    solver.Print();

    return 0;
}

// Complex symmetric example
int main_complex() {
    const int n = 1000;
    const int nnz = 5000;

    std::vector<int> Ap(n + 1), Ai(nnz);
    std::vector<F90Complex> Ax(nnz), b(n), x(n);

    // F90Complex is struct { double x, y; } representing real + imag

    // ... fill complex matrix data ...

    BlitBLQMR<F90Complex> solver(n);
    solver.Prepare(Ap.data(), Ai.data(), Ax.data(), nnz);
    solver.Solve(x.data(), b.data(), 1);

    return 0;
}
```

### ILU Preconditioner Class

```cpp
#include "blit_solvers.h"

// Standalone ILU preconditioner (can be used with other solvers)
BlitILU<double> ilu(n, nnz);

// Prepare: compute ILU factorization
double droptol = 1e-3;
ilu.Run(&Ap, &Ai, &Ax, droptol);

// Apply: solve M * y = x (preconditioner application)
ilu.Solve(nrow, ncol, y.data(), x.data());
```

### Compilation

```bash
# Compile with Fortran library
g++ -o myprogram myprogram.cpp \
    -I/path/to/blit/include \
    -L/path/to/blit/lib \
    -lblit -lumfpack -lamd -lblas -llapack -lgfortran

# Or with pkg-config if installed
g++ -o myprogram myprogram.cpp $(pkg-config --cflags --libs blit)
```

---

## Preconditioning Guide

### Quick Recommendations

| Situation | Preconditioner | Code (Python) |
|-----------|---------------|---------------|
| First attempt | None | `blqmr(A, b, precond_type=None)` |
| Poorly scaled diagonal | Jacobi | `blqmr(A, b, precond_type='diag')` |
| General sparse | ILU | `blqmr(A, b, precond_type='ilu')` |
| ILU fails (indefinite) | Jacobi | `blqmr(A, b, precond_type='diag')` |
| Symmetric preservation | Split Jacobi | `blqmr(A, b, M1=sqrt_D, M2=sqrt_D)` |

### Split Preconditioning

For symmetric matrices, **split preconditioning** preserves symmetry:

```
Original:     A x = b
Transformed:  (M₁⁻¹ A M₂⁻¹) y = M₁⁻¹ b
Recover:      x = M₂⁻¹ y
```

With M₁ = M₂ = √D (square root of diagonal), the preconditioned matrix remains symmetric.

**Python:**
```python
d = np.abs(A.diagonal())
d[d < 1e-14] = 1.0
M = sparse.diags(np.sqrt(d))
result = blqmr(A, b, M1=M, M2=M, precond_type=None)
```

**MATLAB:**
```matlab
d = abs(diag(A));
d(d < 1e-14) = 1;
M = spdiags(sqrt(d), 0, n, n);
[x, flag] = blqmr(A, b, tol, maxit, M, M);
```

### Preconditioner Comparison

| Type | Strength | Build | Apply | Parallel | Best For |
|------|----------|-------|-------|----------|----------|
| None | — | O(1) | O(1) | ✅ | Well-conditioned |
| Jacobi | Weak | O(n) | O(n) | ✅ | Scaling issues |
| ILU(0) | Good | O(nnz) | O(nnz) | ❌ | General sparse |
| ILUT | Very Good | O(nnz+) | O(nnz+) | ❌ | Difficult systems |

---

## Benchmarks

### BLQMR vs QMR vs Direct Solver

The following benchmarks compare BLQMR against MATLAB's built-in `qmr()` (point method) and `mldivide` (direct solver using UMFPACK) on tetrahedral FEM matrices.

**Configuration:**
- Grid sizes: 5³ to 50³ nodes (125 to 125,000 unknowns)
- Block size: 4 right-hand sides
- Tolerance: 10⁻⁸
- Preconditioner: Split Jacobi

#### Real Symmetric FEM Matrices

| Grid | Nodes | NNZ | mldivide | QMR | BLQMR | BLQMR vs mldiv | BLQMR vs QMR |
|------|-------|-----|----------|-----|-------|----------------|--------------|
| 10³ | 1,000 | 6,400 | 1.7ms | 4.3ms | 4.6ms | 2.7× slower | 1.1× slower |
| 20³ | 8,000 | 53,600 | 19ms | 38ms | 44ms | 2.3× slower | 1.2× slower |
| 30³ | 27,000 | 183,600 | 117ms | 91ms | 125ms | 1.1× slower | 1.4× slower |
| 40³ | 64,000 | 438,400 | 517ms | 176ms | 278ms | **1.9× faster** | 1.6× slower |
| 50³ | 125,000 | 860,000 | 1.95s | 311ms | 516ms | **3.8× faster** | 1.7× slower |

#### Complex Symmetric FEM Matrices

| Grid | Nodes | NNZ | mldivide | QMR | BLQMR | BLQMR vs mldiv | BLQMR vs QMR |
|------|-------|-----|----------|-----|-------|----------------|--------------|
| 10³ | 1,000 | 12,718 | 6.3ms | 25ms | 19ms | 3.1× slower | **1.3× faster** |
| 20³ | 8,000 | 110,638 | 135ms | 143ms | 115ms | **1.2× faster** | **1.2× faster** |
| 30³ | 27,000 | 383,758 | 1.36s | 429ms | 373ms | **3.6× faster** | **1.2× faster** |
| 40³ | 64,000 | 922,078 | 6.40s | 980ms | 947ms | **6.8× faster** | 1.03× faster |
| 50³ | 125,000 | 1,815,598 | 25.9s | 1.85s | 1.76s | **14.7× faster** | 1.05× faster |

**Key Observations:**
- For **large complex systems** (n > 20,000), BLQMR is **3-15× faster** than direct solvers
- BLQMR achieves **near-ideal iteration scaling**: with 4 RHS, iterations are ~24% of 4× single solves
- Residual accuracy is comparable across all methods (~10⁻⁸ to 10⁻⁹)

#### Iteration Efficiency

| Grid | Nodes | QMR (4×single) | BLQMR (block) | Ratio | Ideal (1/4) |
|------|-------|----------------|---------------|-------|-------------|
| 10³ | 1,000 | 332 | 58 | 0.175 | 0.250 |
| 20³ | 8,000 | 393 | 84 | 0.214 | 0.250 |
| 30³ | 27,000 | 419 | 92 | 0.220 | 0.250 |
| 50³ | 125,000 | 418 | 99 | 0.237 | 0.250 |

BLQMR achieves **super-linear speedup** in iterations—better than the theoretical 4× reduction!

### Block Size Analysis

How does performance scale with different block sizes? (64 RHS total, n=8000 complex symmetric)

| Block Size | Iterations | Time (s) | Speedup vs QMR | Iteration Ratio |
|------------|------------|----------|----------------|-----------------|
| 1 (point) | 10,154 | 5.00 | 1.31× | 1.000 |
| 4 | 2,220 | 3.58 | 1.83× | 0.219 |
| 8 | 956 | 3.30 | 1.98× | 0.094 |
| 16 | 361 | 3.11 | 2.10× | 0.036 |
| 32 | 178 | 3.02 | 2.16× | 0.018 |
| 64 | 51 | 4.05 | 1.62× | 0.005 |

**Optimal block size**: 16-32 for this problem. Too large blocks (64) increase per-iteration cost without proportional iteration reduction.

**Recommendation**: Start with block size 4-16, adjust based on:
- Memory constraints (larger blocks need more workspace)
- Iteration reduction (diminishing returns beyond ~32)
- Per-iteration cost (dense operations scale as O(m²) with block size m)

---

## Performance Optimization

### 1. Use the Fortran Backend (10-100× faster)

```python
from blocksolver import BLQMR_EXT
if not BLQMR_EXT:
    print("Warning: Using slow Python backend")
    print("Install Fortran dependencies for 10-100x speedup")
```

### 2. Batch Multiple Right-Hand Sides

```python
# FAST: Single call with all RHS (shares Krylov information)
B = np.column_stack([b1, b2, ..., b64])
result = blqmr(A, B)

# SLOW: Separate calls (no information sharing)
for b in [b1, b2, ...]:
    result = blqmr(A, b)
```

### 3. Choose Optimal Block Size

Based on benchmarks, block sizes of **8-16** typically offer the best time-to-solution:

```matlab
opt.blocksize = 16;  % Process 16 RHS at a time
[X, flag] = blqmr(A, B, tol, maxit, [], [], [], opt);
```

### 4. Reuse Workspace (Python)

```python
from blocksolver import BLQMRWorkspace

ws = BLQMRWorkspace(n, m, dtype=np.complex128)
for A_k in varying_matrices:
    result = blqmr(A_k, b, workspace=ws)
```

### 5. Tune ILU Drop Tolerance

```python
# Stronger preconditioner (more memory, fewer iterations)
result = blqmr(A, b, droptol=1e-4)

# Weaker preconditioner (less memory, more iterations)
result = blqmr(A, b, droptol=1e-2)
```

---

## Troubleshooting

### "Fortran backend not available"

```bash
# Check status
python -c "from blocksolver import get_backend_info; print(get_backend_info())"

# Install dependencies
sudo apt install gfortran libsuitesparse-dev  # Linux
brew install gcc suite-sparse                  # macOS

# Rebuild
pip install --force-reinstall blocksolver
```

### Slow Convergence

1. Enable preconditioning: `precond_type='ilu'`
2. Check diagonal scaling: if max/min > 100, use Jacobi
3. Reduce tolerance gradually to identify stagnation point

### ILU Factorization Fails

For indefinite matrices:
```python
# Fall back to diagonal preconditioner
result = blqmr(A, b, precond_type='diag')
```

### Stagnation (flag=3)

```python
# Use true residual monitoring
result = blqmr(A, b, residual=True)

# Verify with actual computation
true_res = np.linalg.norm(A @ result.x - b) / np.linalg.norm(b)
```

---

## Algorithm Details

### Block Quasi-Minimal Residual

BLIT implements the algorithm from Boyse & Seidl (1996), extended for complex symmetric systems using the quasi inner product.

**Key components:**
1. **Quasi-QR decomposition**: Modified Gram-Schmidt with ⟨x,y⟩ = Σ xₖyₖ
2. **Three-term Lanczos recurrence**: Memory-efficient basis construction
3. **Block updates**: Shared matrix-vector products across all RHS

**Cost per iteration:**
- 1 sparse matrix-vector multiply (shared)
- 2 preconditioner applications
- O(nm²) dense operations for m RHS

---

## Citation

```bibtex
@article{fang2004microwave,
  title={Microwave image reconstruction from 3-D fields coupled to 2-D parameter estimation},
  author={Fang, Qianqian and Meaney, Paul M and Geimer, Steven D and others},
  journal={IEEE Transactions on Medical Imaging},
  volume={23}, number={4}, pages={475--484}, year={2004}
}

@article{boyse1996block,
  title={A block QMR method for computing multiple simultaneous solutions},
  author={Boyse, William E and Seidl, Andrew A},
  journal={SIAM Journal on Scientific Computing},
  volume={17}, number={1}, pages={263--274}, year={1996}
}
```

---

## License

BSD-3-Clause or GPL-3.0+ (dual-licensed)

---

## Links

- **Repository**: https://github.com/fangq/blit
- **PyPI**: https://pypi.org/project/blocksolver
- **Issues**: https://github.com/fangq/blit/issues
