# BLIT Python Bindings

Python interface for the BLIT (Block Iterative) sparse linear solver library.

## Installation

### Prerequisites

- Python >= 3.8
- NumPy
- Fortran compiler (gfortran, ifort)
- UMFPACK/SuiteSparse library
- BLAS/LAPACK

On Ubuntu/Debian:
```bash
sudo apt install gfortran libsuitesparse-dev libblas-dev liblapack-dev
```

On macOS (Homebrew):
```bash
brew install gcc suite-sparse openblas
```

### Install

```bash
cd python
pip install .
```

For development:
```bash
pip install -e .
```

## Usage

### Basic Usage

```python
import numpy as np
from blit import blqmr_solve

# Define sparse matrix in CSC format (0-based indexing)
Ap = np.array([0, 2, 5, 9, 10, 12], dtype=np.int32)
Ai = np.array([0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4], dtype=np.int32)
Ax = np.array([2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1.])
b = np.array([8.0, 45.0, -3.0, 3.0, 19.0])

# Solve
result = blqmr_solve(Ap, Ai, Ax, b, tol=1e-8)

print(f"Solution: {result.x}")
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iter}")
```

### With SciPy Sparse Matrices

```python
from scipy.sparse import csc_matrix
from blit import blqmr_scipy

A = csc_matrix([[4, 1, 0], [1, 3, 1], [0, 1, 2]])
b = np.array([1., 2., 3.])

x, flag = blqmr_scipy(A, b, tol=1e-10)
```

### Multiple Right-Hand Sides

```python
from blit import blqmr_solve_multi

B = np.column_stack([b1, b2, b3])  # n x nrhs
result = blqmr_solve_multi(Ap, Ai, Ax, B)
# result.x is n x nrhs
```

## API Reference

### `blqmr_solve(Ap, Ai, Ax, b, **kwargs) -> BLQMRResult`

Solve sparse system Ax = b.

**Parameters:**
- `Ap`: Column pointers (int32, length n+1)
- `Ai`: Row indices (int32, length nnz)  
- `Ax`: Non-zero values (float64, length nnz)
- `b`: Right-hand side (float64, length n)
- `tol`: Convergence tolerance (default: 1e-6)
- `maxiter`: Maximum iterations (default: n)
- `droptol`: ILU drop tolerance (default: 0.001)
- `use_precond`: Use ILU preconditioner (default: True)
- `zero_based`: Input uses 0-based indexing (default: True)

**Returns:** `BLQMRResult` with attributes:
- `x`: Solution vector
- `flag`: 0=converged, 1=maxiter, 2=precond fail, 3=stagnation
- `iter`: Iterations performed
- `relres`: Relative residual
- `converged`: Boolean property

## Testing

```bash
make test
# or
pytest tests/ -v
```

## License

BSD / LGPL / GPL - see LICENSE files in parent directory.
