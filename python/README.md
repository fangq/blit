# BlockSolver - Block Quasi-Minimal Residual (BLQMR) Sparse Linear Solver

- Copyright: (C) Qianqian Fang (2005, 2011, 2026) <q.fang at neu.edu>
- License: BSD-3-Clause and GPL-v3 dual-licensed
- Version: 0.9.4
- Website: https://neurojson.org/Page/blocksolver
- Github: https://github.com/fangq/blit
- PyPi Page: https://pypi.org/project/blocksolver/

**BlockSolver** is a Python package for solving large sparse linear systems using the Block Quasi-Minimal Residual (BLQMR) algorithm. It provides both a high-performance Fortran backend and a pure Python/NumPy implementation for maximum portability.

## Features

- **Block QMR Algorithm**: Efficiently solves multiple right-hand sides simultaneously
- **Complex Symmetric Support**: Designed for complex symmetric matrices (A = Aᵀ, not A = A†)
- **Dual Backend**: Fortran extension for speed, Python fallback for portability
- **Flexible Preconditioning**: ILU, diagonal (Jacobi), and split preconditioners
- **SciPy Integration**: Works seamlessly with SciPy sparse matrices
- **Optional Numba Acceleration**: JIT-compiled kernels for the Python backend

## Algorithm

### Block Quasi-Minimal Residual (BLQMR)

The BLQMR algorithm is an iterative Krylov subspace method specifically designed for:

1. **Complex symmetric systems**: Unlike standard methods that assume Hermitian (A = A†) or general matrices, BLQMR exploits complex symmetry (A = Aᵀ) which arises in electromagnetics, acoustics, and diffuse optical tomography.

2. **Multiple right-hand sides**: Instead of solving each system independently, BLQMR processes all right-hand sides together in a block fashion, sharing Krylov subspace information and reducing total computation.

3. **Quasi-minimal residual**: The algorithm minimizes a quasi-residual norm at each iteration, providing smooth convergence without the erratic behavior of some Krylov methods.

### Key Components

- **Quasi-QR Decomposition**: A modified Gram-Schmidt process using the quasi inner product ⟨x,y⟩ = Σ xₖyₖ (without conjugation) for complex symmetric systems.

- **Three-term Lanczos Recurrence**: Builds an orthonormal basis for the Krylov subspace with short recurrences, minimizing memory usage.

- **Block Updates**: Processes m right-hand sides simultaneously, with typical block sizes of 1-64.

### When to Use BLQMR

| Use Case | Recommendation |
|----------|----------------|
| Complex symmetric matrix (A = Aᵀ) | ✅ Ideal |
| Multiple right-hand sides | ✅ Ideal |
| Real symmetric positive definite | Consider CG first |
| General non-symmetric | Consider GMRES or BiCGSTAB |
| Very large systems (>10⁶ unknowns) | ✅ Good with preconditioning |

## Installation

### From PyPI

```bash
pip install blocksolver
```

### From Source

Prerequisites:
- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.0
- (Optional) Fortran compiler + UMFPACK for the accelerated backend
- (Optional) Numba for accelerated Python backend

```bash
# Ubuntu/Debian
sudo apt install gfortran libsuitesparse-dev libblas-dev liblapack-dev

# macOS
brew install gcc suite-sparse openblas

# Install
cd python
pip install .
```

## Quick Start

```python
import numpy as np
from scipy.sparse import csc_matrix
from blocksolver import blqmr

# Create a sparse matrix
A = csc_matrix([
    [4, 1, 0, 0],
    [1, 4, 1, 0],
    [0, 1, 4, 1],
    [0, 0, 1, 4]
], dtype=float)

b = np.array([1., 2., 3., 4.])

# Solve Ax = b
result = blqmr(A, b, tol=1e-10)

print(f"Solution: {result.x}")
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iter}")
print(f"Relative residual: {result.relres:.2e}")
```

## Usage

### Main Interface: `blqmr()`

The primary function `blqmr()` automatically selects the best available backend (Fortran if available, otherwise Python).

```python
from blocksolver import blqmr, BLQMR_EXT

# Check which backend is active
print(f"Using Fortran backend: {BLQMR_EXT}")

# Basic usage
result = blqmr(A, b)

# With options
result = blqmr(A, b, 
    tol=1e-8,              # Convergence tolerance
    maxiter=1000,          # Maximum iterations
    precond_type='ilu',    # Preconditioner: 'ilu', 'diag', or None
)
```

### Multiple Right-Hand Sides

BLQMR excels when solving the same system with multiple right-hand sides:

```python
import numpy as np
from blocksolver import blqmr

# 100 different right-hand sides
B = np.random.randn(n, 100)

# Solve all systems at once (much faster than solving individually)
result = blqmr(A, B, tol=1e-8)

# result.x has shape (n, 100)
```

### Complex Symmetric Systems

BLQMR is specifically designed for complex symmetric matrices (common in frequency-domain wave problems):

```python
import numpy as np
from blocksolver import blqmr

# Complex symmetric matrix (A = A.T, NOT A.conj().T)
A = create_helmholtz_matrix(frequency=1000)  # Your application
b = np.complex128(source_term)

result = blqmr(A, b, tol=1e-8, precond_type='diag')
```

### Preconditioning

BlockSolver supports multiple preconditioner types for both backends:

```python
from blocksolver import blqmr, make_preconditioner

# Using precond_type parameter (works with both backends)
result = blqmr(A, b, precond_type='ilu')    # Incomplete LU
result = blqmr(A, b, precond_type='diag')   # Diagonal (Jacobi)
result = blqmr(A, b, precond_type=None)     # No preconditioning

# Custom preconditioner (Python backend only)
M1 = make_preconditioner(A, 'ilu', drop_tol=1e-4, fill_factor=10)
result = blqmr(A, b, M1=M1, precond_type=None)

# Split preconditioning for symmetric systems (Python backend)
# Preserves symmetry: M1^{-1} A M2^{-1}
M = make_preconditioner(A, 'diag', split=True)  # Returns sqrt(D)
result = blqmr(A, b, M1=M, M2=M, precond_type=None)
```

### SciPy-Compatible Interface

For drop-in replacement in existing code:

```python
from blocksolver import blqmr_scipy

# Returns (x, flag) like scipy.sparse.linalg solvers
x, flag = blqmr_scipy(A, b, tol=1e-10)
```

### Low-Level CSC Interface

For maximum control, use the CSC component interface:

```python
from blocksolver import blqmr_solve

# CSC format components (0-based indexing)
Ap = np.array([0, 2, 5, 9, 10, 12], dtype=np.int32)  # Column pointers
Ai = np.array([0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4], dtype=np.int32)  # Row indices
Ax = np.array([2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1.])  # Values
b = np.array([8., 45., -3., 3., 19.])

result = blqmr_solve(Ap, Ai, Ax, b, 
    tol=1e-8,
    droptol=0.001,         # ILU drop tolerance (Fortran backend only)
    precond_type='ilu',    # Preconditioner type
    zero_based=True,       # 0-based indexing (default)
)
```

## API Reference

### `blqmr(A, B, **kwargs) -> BLQMRResult`

Main solver interface.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | sparse matrix or ndarray | required | System matrix (n × n) |
| `B` | ndarray | required | Right-hand side (n,) or (n × m) |
| `tol` | float | 1e-6 | Convergence tolerance |
| `maxiter` | int | n | Maximum iterations |
| `M1`, `M2` | preconditioner | None | Custom preconditioners (Python backend) |
| `x0` | ndarray | None | Initial guess |
| `precond_type` | str or None | 'ilu' | Preconditioner: 'ilu', 'diag', or None |
| `droptol` | float | 0.001 | ILU drop tolerance (Fortran backend) |
| `residual` | bool | False | Use true residual for convergence (Python) |
| `workspace` | BLQMRWorkspace | None | Pre-allocated workspace (Python) |

**Returns:** `BLQMRResult` object with:
| Attribute | Type | Description |
|-----------|------|-------------|
| `x` | ndarray | Solution vector(s) |
| `flag` | int | 0=converged, 1=maxiter, 2=precond fail, 3=stagnation |
| `iter` | int | Iterations performed |
| `relres` | float | Final relative residual |
| `converged` | bool | True if flag == 0 |
| `resv` | ndarray | Residual history (Python backend only) |

### `blqmr_solve(Ap, Ai, Ax, b, **kwargs) -> BLQMRResult`

Low-level CSC interface for single RHS.

### `blqmr_solve_multi(Ap, Ai, Ax, B, **kwargs) -> BLQMRResult`

Low-level CSC interface for multiple right-hand sides.

### `blqmr_scipy(A, b, **kwargs) -> Tuple[ndarray, int]`

SciPy-compatible interface returning `(x, flag)`.

### `make_preconditioner(A, precond_type, **kwargs) -> Preconditioner`

Create a preconditioner for the Python backend.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A` | sparse matrix | required | System matrix |
| `precond_type` | str | required | 'diag', 'jacobi', 'ilu', 'ilu0', 'ilut', 'lu', 'ssor' |
| `split` | bool | False | Return sqrt(D) for split preconditioning |
| `drop_tol` | float | 1e-4 | Drop tolerance for ILUT |
| `fill_factor` | float | 10 | Fill factor for ILUT |
| `omega` | float | 1.0 | Relaxation parameter for SSOR |

### Utility Functions

```python
from blocksolver import (
    BLQMR_EXT,        # True if Fortran backend available
    HAS_NUMBA,        # True if Numba acceleration available
    get_backend_info, # Returns dict with backend details
    test,             # Run built-in tests
)
```

## Benchmarks

### BLQMR vs Direct Solver (mldivide)

Complex symmetric FEM matrices, 4 right-hand sides, tolerance 10⁻⁸, split Jacobi preconditioner:

| Grid | Nodes | NNZ | mldivide | BLQMR | Speedup |
|------|-------|-----|----------|-------|---------|
| 20³ | 8,000 | 110K | 135ms | 115ms | **1.2×** |
| 30³ | 27,000 | 384K | 1.36s | 373ms | **3.6×** |
| 40³ | 64,000 | 922K | 6.40s | 947ms | **6.8×** |
| 50³ | 125,000 | 1.8M | 25.9s | 1.76s | **14.7×** |

### Block Size Efficiency

With 64 RHS on a 8,000-node complex symmetric system:

| Block Size | Iterations | Speedup vs Single |
|------------|------------|-------------------|
| 1 (point) | 10,154 | 1.0× |
| 4 | 2,220 | 1.8× |
| 8 | 956 | 2.0× |
| 16 | 361 | 2.1× |
| 32 | 178 | 2.2× |

**Optimal block size**: 8-16 for most problems. Larger blocks have diminishing returns due to increased per-iteration cost.

### Iteration Efficiency

With 4 RHS, BLQMR uses only ~24% of total iterations compared to 4 separate single-RHS solves — achieving **super-linear block acceleration**.

## Performance Tips

1. **Use the Fortran backend** when available (faster for large systems)

2. **Enable preconditioning** for ill-conditioned systems:
   ```python
   result = blqmr(A, b, precond_type='ilu')
   ```

3. **Batch multiple right-hand sides** instead of solving one at a time:
   ```python
   # Fast: single call with all RHS
   result = blqmr(A, B_matrix)
   
   # Slow: multiple calls
   for b in B_columns:
       result = blqmr(A, b)
   ```

4. **Install Numba** for faster Python backend:
   ```bash
   pip install numba
   ```

5. **Reuse workspace** for repeated solves with the same dimensions:
   ```python
   from blocksolver import BLQMRWorkspace
   ws = BLQMRWorkspace(n, m, dtype=np.complex128)
   for b in many_rhs:
       result = blqmr(A, b, workspace=ws)
   ```

6. **Use split Jacobi for complex symmetric systems**:
   ```python
   # Preserves symmetry of preconditioned system
   M = make_preconditioner(A, 'diag', split=True)
   result = blqmr(A, b, M1=M, M2=M, precond_type=None)
   ```

## Examples

### Diffuse Optical Tomography

```python
import numpy as np
from scipy.sparse import diags, kron, eye
from blocksolver import blqmr

def create_diffusion_matrix(nx, ny, D=1.0, mu_a=0.01, omega=1e9):
    """Create 2D diffusion matrix for DOT."""
    n = nx * ny
    h = 1.0 / nx
    
    # Laplacian
    Lx = diags([-1, 2, -1], [-1, 0, 1], shape=(nx, nx)) / h**2
    Ly = diags([-1, 2, -1], [-1, 0, 1], shape=(ny, ny)) / h**2
    L = kron(eye(ny), Lx) + kron(Ly, eye(nx))
    
    # Diffusion equation: (-D∇² + μ_a + iω/c) φ = q
    c = 3e10  # speed of light in tissue (cm/s)
    A = -D * L + mu_a * eye(n) + 1j * omega / c * eye(n)
    
    return A.tocsc()

# Setup problem
A = create_diffusion_matrix(100, 100, omega=2*np.pi*100e6)
sources = np.random.randn(10000, 16) + 0j  # 16 source positions

# Solve for all sources at once
result = blqmr(A, sources, tol=1e-8, precond_type='diag')
print(f"Solved {sources.shape[1]} systems in {result.iter} iterations")
```

### Frequency-Domain Acoustics

```python
import numpy as np
from blocksolver import blqmr

# Helmholtz equation: (∇² + k²)p = f
# Results in complex symmetric matrix

def solve_helmholtz(K, M, f, frequencies):
    """Solve Helmholtz at multiple frequencies."""
    solutions = []
    for omega in frequencies:
        # A = K - ω²M (complex symmetric if K, M are symmetric)
        A = K - omega**2 * M
        result = blqmr(A, f, tol=1e-10, precond_type='diag')
        solutions.append(result.x)
    return np.array(solutions)
```

## Troubleshooting

### "No Fortran backend available"

Install the package with Fortran support:
```bash
# Install dependencies first
sudo apt install gfortran libsuitesparse-dev  # Linux
brew install gcc suite-sparse                  # macOS

# Reinstall blocksolver
pip install --no-cache-dir blocksolver
```

### Check backend status

```python
from blocksolver import get_backend_info
print(get_backend_info())
# {'backend': 'binary', 'has_fortran': True, 'has_numba': True}
```

### Slow convergence

1. Enable preconditioning: `precond_type='ilu'` or `precond_type='diag'`
2. Reduce ILU drop tolerance: `droptol=1e-4` (Fortran backend)
3. Check matrix conditioning with `np.linalg.cond(A.toarray())`

### ILU factorization fails

For indefinite or complex symmetric matrices, ILU may fail:
```python
# Fall back to diagonal preconditioner
result = blqmr(A, b, precond_type='diag')
```

### Memory issues with large systems

1. Use the Fortran backend (more memory efficient)
2. Reduce block size for multiple RHS
3. Use iterative refinement instead of tighter tolerance

## License

BSD-3-Clause or GPL-3.0+ (dual-licensed)

## Citation

If you use BlockSolver in your research, please cite:

```bibtex
@software{blocksolver,
  author = {Qianqian Fang},
  title = {BlockSolver: Block Quasi-Minimal Residual Sparse Linear Solver},
  url = {https://github.com/fangq/blit},
  year = {2024}
}
```

## See Also

- [BLIT](https://github.com/fangq/blit) - The underlying Fortran library
- [SciPy sparse.linalg](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html) - Other iterative solvers
- [PyAMG](https://github.com/pyamg/pyamg) - Algebraic multigrid solvers