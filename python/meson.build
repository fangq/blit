# Meson build for BLIT Python extension
#
# Run from python/ directory:
#   pip install . --no-build-isolation
#   # or
#   meson setup builddir && meson compile -C builddir

project('blit', ['c', 'fortran'],
  version: '1.0.0',
  license: 'BSD-3-Clause OR LGPL-3.0-or-later',
  meson_version: '>=0.64.0',
  default_options: ['buildtype=release', 'fortran_std=f2008']
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# NumPy includes
incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'], check: true
).stdout().strip()

incdir_f2py = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'], check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

# Source directory (relative to this meson.build)
src_dir = '../src'

# Find dependencies
cc = meson.get_compiler('c')

umfpack_dep = dependency('umfpack', required: false)
if not umfpack_dep.found()
  umfpack_lib = cc.find_library('umfpack', required: true)
  amd_lib = cc.find_library('amd', required: true)
  cholmod_lib = cc.find_library('cholmod', required: false)
  colamd_lib = cc.find_library('colamd', required: false)
  suitesparse_lib = cc.find_library('suitesparseconfig', required: false)

  umfpack_deps = [umfpack_lib, amd_lib]
  foreach lib : [cholmod_lib, colamd_lib, suitesparse_lib]
    if lib.found()
      umfpack_deps += lib
    endif
  endforeach
else
  umfpack_deps = [umfpack_dep]
endif

blas_dep = dependency('blas', required: true)
lapack_dep = dependency('lapack', required: true)

# Sources
fortran_src = files(
  src_dir / 'blit_const.f90',
  src_dir / 'blit_matrixutil.f90',
  src_dir / 'blit_sparseutil.f90',
  src_dir / 'blit_ilupcond.f90',
  src_dir / 'blit_blqmr.f90',
  src_dir / 'blit_blqmr_f2py.f90',
)

c_src = files(src_dir / 'umf4_f77wrapper.c')

# f2py wrapper generation
f2py_wrapper = custom_target('_blqmr_f2py',
  input: src_dir / 'blit_blqmr_f2py.f90',
  output: ['_blqmrmodule.c', '_blqmr-f2pywrappers2.f90'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', '_blqmr',
            '--lower', '--build-dir', '@OUTDIR@']
)

# Build extension
py.extension_module('_blqmr',
  [fortran_src, c_src, f2py_wrapper],
  include_directories: inc_np,
  dependencies: [py_dep, blas_dep, lapack_dep] + umfpack_deps,
  fortran_args: ['-O3', '-cpp', '-fPIC'],
  c_args: ['-O3', '-fPIC'],
  install: true,
  subdir: 'blit',
)

# Install Python files
py.install_sources(
  'blit/__init__.py',
  'blit/blqmr.py',
  subdir: 'blit',
)
