"""
Setup script for BLIT BLQMR Python extension.

Build commands (run from python/ directory):
    pip install .              # Install
    pip install -e .           # Editable install
    python setup.py build_ext --inplace
"""

import os
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).parent.absolute()
ROOT_DIR = PYTHON_DIR.parent
SRC_DIR = ROOT_DIR / "src"

print(f"[setup.py] PYTHON_DIR: {PYTHON_DIR}")
print(f"[setup.py] ROOT_DIR: {ROOT_DIR}")
print(f"[setup.py] SRC_DIR: {SRC_DIR}")
print(f"[setup.py] Python version: {sys.version_info.major}.{sys.version_info.minor}")

# Fortran library sources (order matters)
FORTRAN_SOURCES = [
    "blit_const.f90",
    "blit_matrixutil.f90",
    "blit_sparseutil.f90",
    "blit_ilupcond.f90",
    "blit_blqmr.f90",
    "blit_blqmr_f2py.f90",  # F2PY wrapper with standalone subroutines
]

C_SOURCES = [
    "umf4_f77wrapper.c",
    "blit_blas_threads.c",
]

# F2PY signature file - explicitly defines the Python interface
SIGNATURE_FILE = "blit_blqmr.pyf"


def find_umfpack():
    """Find UMFPACK library paths."""
    include_dirs = [str(SRC_DIR)]
    library_dirs = []
    libraries = [
        "umfpack",
        "amd",
        "cholmod",
        "colamd",
        "suitesparseconfig",
        "blas",
        "lapack",
    ]

    search_paths = [
        "/usr/local",
        "/usr",
        "/opt/homebrew",
        "/opt/local",
        os.path.expanduser("~/.local"),
    ]

    if "CONDA_PREFIX" in os.environ:
        search_paths.insert(0, os.environ["CONDA_PREFIX"])

    for base in search_paths:
        for inc in ["include/suitesparse", "include"]:
            p = os.path.join(base, inc)
            if os.path.exists(p) and p not in include_dirs:
                include_dirs.append(p)
        for lib in ["lib", "lib64"]:
            p = os.path.join(base, lib)
            if os.path.exists(p) and p not in library_dirs:
                library_dirs.append(p)

    for var, lst in [("UMFPACK_INCLUDE", include_dirs), ("UMFPACK_LIB", library_dirs)]:
        if var in os.environ:
            for p in os.environ[var].split(os.pathsep):
                if p and p not in lst:
                    lst.insert(0, p)

    if "SUITESPARSE_ROOT" in os.environ:
        root = os.environ["SUITESPARSE_ROOT"]
        include_dirs.insert(0, os.path.join(root, "include"))
        library_dirs.insert(0, os.path.join(root, "lib"))

    print(f"[setup.py] include_dirs: {include_dirs}")
    print(f"[setup.py] library_dirs: {library_dirs}")
    print(f"[setup.py] libraries: {libraries}")

    return include_dirs, library_dirs, libraries


def check_source_files():
    """Check if all source files exist and return list of missing files."""
    missing = []

    # Check .pyf file
    pyf_path = PYTHON_DIR / SIGNATURE_FILE
    print(f"[setup.py] Checking {pyf_path}: {pyf_path.exists()}")
    if not pyf_path.exists():
        missing.append(str(pyf_path))

    # Check Fortran sources
    for f in FORTRAN_SOURCES:
        p = SRC_DIR / f
        print(f"[setup.py] Checking {p}: {p.exists()}")
        if not p.exists():
            missing.append(str(p))

    # Check C sources
    for f in C_SOURCES:
        p = SRC_DIR / f
        print(f"[setup.py] Checking {p}: {p.exists()}")
        if not p.exists():
            missing.append(str(p))

    return missing


# Try numpy.distutils first (NumPy < 2.0 and Python < 3.12), fall back to setuptools
USE_NUMPY_DISTUTILS = False
try:
    # numpy.distutils is deprecated in Python 3.12+ even in NumPy 1.x
    if sys.version_info < (3, 12):
        from numpy.distutils.core import setup, Extension

        USE_NUMPY_DISTUTILS = True
        print("[setup.py] Using numpy.distutils")
    else:
        from setuptools import setup, Extension

        print("[setup.py] Using setuptools (Python 3.12+)")
except ImportError as e:
    from setuptools import setup, Extension

    print(f"[setup.py] Using setuptools (numpy.distutils import failed: {e})")


def get_extension():
    """Create the Fortran extension."""
    include_dirs, library_dirs, libraries = find_umfpack()

    # Source list: signature file FIRST, then Fortran sources, then C
    # The .pyf file tells f2py exactly what to wrap
    sources = [str(PYTHON_DIR / SIGNATURE_FILE)]
    sources += [str(SRC_DIR / f) for f in FORTRAN_SOURCES]
    sources += [str(SRC_DIR / f) for f in C_SOURCES]

    print(f"[setup.py] Extension sources: {sources}")

    return Extension(
        name="blocksolver._blqmr",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_f90_compile_args=["-O3", "-fPIC", "-cpp", "-fopenmp"]
        if USE_NUMPY_DISTUTILS
        else [],
        extra_f77_compile_args=["-O3", "-fPIC", "-fopenmp"]
        if USE_NUMPY_DISTUTILS
        else [],
        extra_compile_args=["-O3", "-fPIC"],
        extra_link_args=["-fopenmp", "-static-libgcc"],
    )


def run_setup_with_extension():
    """Run setup with Fortran extension."""
    print("[setup.py] Running setup WITH Fortran extension")
    setup(
        name="blocksolver",
        version="0.9.0",
        description="Block Quasi-Minimal-Residual sparse linear solver",
        author="Qianqian Fang",
        author_email="q.fang@neu.edu",
        url="https://neurojson.org/Page/blocksolver",
        license="BSD/LGPL/GPL",
        packages=["blocksolver"],
        ext_modules=[get_extension()],
        python_requires=">=3.8",
        install_requires=[
            "numpy>=1.16",
            "scipy>=1.0",
        ],
        extras_require={
            "fast": ["numba>=0.50"],
            "test": ["pytest>=6.0"],
        },
    )


def run_setup_without_extension():
    """Run setup without Fortran extension (pure Python fallback)."""
    print("[setup.py] Running setup WITHOUT Fortran extension (pure Python only)")
    setup(
        name="blocksolver",
        version="0.9.0",
        description="Block Quasi-Minimal-Residual sparse linear solver",
        author="Qianqian Fang",
        author_email="q.fang@neu.edu",
        url="https://neurojson.org/Page/blocksolver",
        license="BSD/LGPL/GPL",
        packages=["blocksolver"],
        python_requires=">=3.8",
        install_requires=[
            "numpy>=1.16",
            "scipy>=1.0",
        ],
        extras_require={
            "fast": ["numba>=0.50"],
            "test": ["pytest>=6.0"],
        },
    )


if __name__ == "__main__":
    print("=" * 60)
    print("[setup.py] Starting build")
    print("=" * 60)

    # Check for --pure-python flag to force pure Python build
    force_pure_python = "--pure-python" in sys.argv
    if force_pure_python:
        sys.argv.remove("--pure-python")
        print("[setup.py] --pure-python flag detected, forcing pure Python build")
        run_setup_without_extension()
        sys.exit(0)

    # Check if source files exist
    missing_files = check_source_files()
    if missing_files:
        print(f"[setup.py] ERROR: Missing source files: {missing_files}")
        print("[setup.py] Cannot build Fortran extension, falling back to pure Python")
        run_setup_without_extension()
        sys.exit(0)

    # Check if we can use numpy.distutils
    if not USE_NUMPY_DISTUTILS:
        if sys.version_info >= (3, 12):
            print("[setup.py] Python 3.12+ detected, numpy.distutils not available")
            print("[setup.py] Use meson build for Fortran extension on Python 3.12+")
        else:
            print("[setup.py] numpy.distutils not available")
        print("[setup.py] Building pure Python version")
        run_setup_without_extension()
        sys.exit(0)

    # Try to build with extension
    print("[setup.py] All source files found, attempting to build Fortran extension")
    try:
        run_setup_with_extension()
    except SystemExit as e:
        # setup() calls sys.exit(), re-raise it
        raise
    except Exception as e:
        print(f"[setup.py] ERROR: Failed to build Fortran extension: {e}")
        import traceback

        traceback.print_exc()
        print("[setup.py] Falling back to pure Python installation")
        run_setup_without_extension()
