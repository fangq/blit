"""Quick diagnostic to verify OpenMP BLQMR path is active."""

import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from blocksolver import BLQMR_EXT, BLQMR_OMP

print(f"BLQMR_EXT (Fortran available): {BLQMR_EXT}")
print(f"BLQMR_OMP (OMP wrappers found): {BLQMR_OMP}")

if BLQMR_EXT:
    from blocksolver import _blqmr

    print(f"\nFortran module functions:")
    for name in sorted(dir(_blqmr)):
        if not name.startswith("_"):
            print(f"  {name}")

    has_real_omp = hasattr(_blqmr, "blqmr_solve_real_multi_omp")
    has_complex_omp = hasattr(_blqmr, "blqmr_solve_complex_multi_omp")
    print(f"\n  blqmr_solve_real_multi_omp:    {has_real_omp}")
    print(f"  blqmr_solve_complex_multi_omp: {has_complex_omp}")

    # Check blqmr.py routing logic
    print(f"\nChecking blqmr.py routing logic:")
    from blocksolver.blqmr import _blqmr_fortran
    import inspect

    src = inspect.getsource(_blqmr_fortran)
    has_nblock_param = "nblock" in inspect.signature(_blqmr_fortran).parameters
    has_omp_call = "multi_omp" in src
    print(f"  _blqmr_fortran has 'nblock' parameter: {has_nblock_param}")
    print(f"  _blqmr_fortran routes to *_multi_omp:  {has_omp_call}")

    if not has_nblock_param:
        print("\n  *** WARNING: _blqmr_fortran does NOT accept nblock!")
        print("  *** The blqmr.py patch for OMP may not be applied.")
        print("  *** Apply the blqmr.py OMP patch and rebuild.")

    if not has_omp_call:
        print("\n  *** WARNING: _blqmr_fortran does NOT call *_multi_omp!")
        print("  *** OMP parallel solve will NOT be used.")

    # Test actual solve path
    if has_nblock_param and has_omp_call and has_complex_omp:
        import numpy as np
        from scipy import sparse

        print(f"\nRunning test solve (n=100, nrhs=4, nblock=1)...")
        n = 100
        np.random.seed(42)
        A = sparse.random(n, n, density=0.1, format="csc", dtype=np.complex128)
        A = A + A.T + 10 * sparse.eye(n, dtype=np.complex128)
        B = np.random.randn(n, 4) + 1j * np.random.randn(n, 4)

        import time

        t0 = time.perf_counter()
        result = _blqmr_fortran(
            A,
            B,
            tol=1e-6,
            maxiter=200,
            x0=None,
            droptol=0.001,
            precond_type=3,
            nblock=1,
        )
        t1 = time.perf_counter()
        print(f"  Time: {t1-t0:.4f}s")
        print(f"  Flag: {result.flag} (0=converged)")
        print(f"  Iter: {result.iter}")
        print(f"  Relres: {result.relres:.2e}")

        # Compare with nblock=0 (no OMP)
        t0 = time.perf_counter()
        result0 = _blqmr_fortran(
            A,
            B,
            tol=1e-6,
            maxiter=200,
            x0=None,
            droptol=0.001,
            precond_type=3,
            nblock=0,
        )
        t1 = time.perf_counter()
        print(
            f"\n  nblock=0 (no OMP): {t1-t0:.4f}s, flag={result0.flag}, iter={result0.iter}"
        )

        if result.flag == 0:
            diff = np.max(np.abs(result.x - result0.x)) / (
                np.max(np.abs(result0.x)) + 1e-300
            )
            print(f"  Solution diff: {diff:.2e}")
            print(
                f"\n  ✓ OMP path is working!"
                if diff < 1e-3
                else f"\n  ✗ Solutions differ too much!"
            )
    else:
        print("\n  Cannot run test — missing OMP support in Fortran or Python.")

else:
    print("\nFortran backend not available — OMP not applicable.")
    print("Build with: cd python && python setup.py build_ext --inplace")
