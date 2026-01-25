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

    return include_dirs, library_dirs, libraries


# Try numpy.distutils first (NumPy < 2.0 and Python < 3.12), fall back to setuptools
USE_NUMPY_DISTUTILS = False
try:
    import sys

    # numpy.distutils is deprecated in Python 3.12+ even in NumPy 1.x
    if sys.version_info < (3, 12):
        from numpy.distutils.core import setup, Extension

        USE_NUMPY_DISTUTILS = True
    else:
        from setuptools import setup, Extension
except ImportError:
    from setuptools import setup, Extension


def get_extension():
    """Create the Fortran extension."""
    include_dirs, library_dirs, libraries = find_umfpack()

    # Source list: signature file FIRST, then Fortran sources, then C
    # The .pyf file tells f2py exactly what to wrap
    sources = [str(PYTHON_DIR / SIGNATURE_FILE)]
    sources += [str(SRC_DIR / f) for f in FORTRAN_SOURCES]
    sources += [str(SRC_DIR / f) for f in C_SOURCES]

    # Check for missing files
    missing = [s for s in sources if not os.path.exists(s)]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")

    return Extension(
        name="blocksolver._blqmr",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_f90_compile_args=["-O3", "-fPIC", "-cpp"] if USE_NUMPY_DISTUTILS else [],
        extra_f77_compile_args=["-O3", "-fPIC"] if USE_NUMPY_DISTUTILS else [],
        extra_compile_args=["-O3", "-fPIC"],
    )


def run_setup_with_extension():
    """Run setup with Fortran extension."""
    setup(
        name="blocksolver",
        version="0.8.1",
        description="Block Quasi-Minimal-Residual sparse linear solver",
        author="Qianqian Fang",
        author_email="q.fang@neu.edu",
        url="https://blit.sourceforge.net",
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
    setup(
        name="blocksolver",
        version="0.8.1",
        description="Block Quasi-Minimal-Residual sparse linear solver",
        author="Qianqian Fang",
        author_email="q.fang@neu.edu",
        url="https://blit.sourceforge.net",
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
    # Check if we can build the extension
    can_build_extension = USE_NUMPY_DISTUTILS

    # Check for required source files
    if can_build_extension:
        try:
            get_extension()
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Building without Fortran extension (pure Python only)")
            can_build_extension = False

    if can_build_extension:
        try:
            run_setup_with_extension()
        except Exception as e:
            print(f"Warning: Failed to build Fortran extension: {e}")
            print("Falling back to pure Python installation")
            run_setup_without_extension()
    else:
        import sys

        if sys.version_info >= (3, 12):
            print("Note: Python 3.12+ does not support numpy.distutils")
            print("Installing pure Python version")
            print("For Fortran extension, use Python 3.11 or earlier with NumPy <2.0")
        elif not USE_NUMPY_DISTUTILS:
            print("Note: numpy.distutils not available")
            print(
                "Installing pure Python version (use NumPy <2.0 for Fortran extension)"
            )
        run_setup_without_extension()
