# Meson build for BLIT Python extension
#
# Run from python/ directory:
#   pip install . 
#   # or
#   pip install . --no-build-isolation
#   # or for development
#   pip install -e . --no-build-isolation

project('blocksolver', ['c', 'fortran'],
  version: '0.8.1',
  license: 'BSD-3-Clause OR LGPL-3.0-or-later',
  meson_version: '>=0.64.0',
  default_options: ['buildtype=release']
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

# f2py fortranobject.c - required for linking f2py extensions
f2py_source_dir = incdir_f2py
fortranobject_c = f2py_source_dir / 'fortranobject.c'

# Source directory (relative to this meson.build)
src_dir = '../src'

# Find compilers
cc = meson.get_compiler('c')
fc = meson.get_compiler('fortran')

# Detect OS
is_windows = host_machine.system() == 'windows'
is_macos = host_machine.system() == 'darwin'

# Platform-specific path separator for environment variables
if is_windows
  path_sep = ';'
else
  path_sep = ':'
endif

# SuiteSparse include directories - check common locations
suitesparse_inc_dirs = []

if is_windows
  # Windows paths (MSYS2/MinGW)
  win_search_paths = [
    'D:/a/_temp/msys64/mingw64/include/suitesparse',
    'D:/a/_temp/msys64/mingw64/include',
    'C:/msys64/mingw64/include/suitesparse',
    'C:/msys64/mingw64/include',
    'C:/tools/msys64/mingw64/include/suitesparse',
    'C:/tools/msys64/mingw64/include',
  ]
  foreach p : win_search_paths
    if run_command('cmd', '/c', 'if exist "' + p + '" echo yes', check: false).stdout().strip() == 'yes'
      suitesparse_inc_dirs += include_directories(p)
    endif
  endforeach
else
  # Unix paths
  unix_search_paths = [
    '/opt/homebrew/include/suitesparse',
    '/usr/local/include/suitesparse',
    '/usr/include/suitesparse',
    '/opt/homebrew/include',
    '/usr/local/include',
  ]
  foreach p : unix_search_paths
    if run_command('test', '-d', p, check: false).returncode() == 0
      suitesparse_inc_dirs += include_directories(p)
    endif
  endforeach
endif

# Check UMFPACK_INCLUDE environment variable
if is_windows
  umfpack_inc_env = run_command('cmd', '/c', 'echo %UMFPACK_INCLUDE%', check: false).stdout().strip()
  if umfpack_inc_env != '%UMFPACK_INCLUDE%' and umfpack_inc_env != ''
    foreach p : umfpack_inc_env.split(path_sep)
      if p != ''
        suitesparse_inc_dirs += include_directories(p)
      endif
    endforeach
  endif
else
  umfpack_inc_env = run_command('sh', '-c', 'echo $UMFPACK_INCLUDE', check: false).stdout().strip()
  if umfpack_inc_env != ''
    foreach p : umfpack_inc_env.split(path_sep)
      if p != '' and run_command('test', '-d', p, check: false).returncode() == 0
        suitesparse_inc_dirs += include_directories(p)
      endif
    endforeach
  endif
endif

# Library search paths
lib_dirs = []

if is_windows
  # Windows library paths
  win_lib_paths = [
    'D:/a/_temp/msys64/mingw64/lib',
    'C:/msys64/mingw64/lib',
    'C:/tools/msys64/mingw64/lib',
  ]
  foreach p : win_lib_paths
    if run_command('cmd', '/c', 'if exist "' + p + '" echo yes', check: false).stdout().strip() == 'yes'
      lib_dirs += p
    endif
  endforeach
  
  # Check UMFPACK_LIB environment variable
  umfpack_lib_env = run_command('cmd', '/c', 'echo %UMFPACK_LIB%', check: false).stdout().strip()
  if umfpack_lib_env != '%UMFPACK_LIB%' and umfpack_lib_env != ''
    foreach p : umfpack_lib_env.split(path_sep)
      if p != ''
        lib_dirs += p
      endif
    endforeach
  endif
else
  # Unix library paths
  umfpack_lib_env = run_command('sh', '-c', 'echo $UMFPACK_LIB', check: false).stdout().strip()
  if umfpack_lib_env != ''
    foreach p : umfpack_lib_env.split(path_sep)
      if p != ''
        lib_dirs += p
      endif
    endforeach
  endif
  
  # Add common Unix library paths
  unix_lib_paths = ['/opt/homebrew/lib', '/usr/local/lib', '/opt/homebrew/opt/openblas/lib']
  foreach p : unix_lib_paths
    if run_command('test', '-d', p, check: false).returncode() == 0
      lib_dirs += p
    endif
  endforeach
endif

# Find UMFPACK and dependencies
umfpack_dep = dependency('umfpack', required: false)
if not umfpack_dep.found()
  umfpack_lib = cc.find_library('umfpack', dirs: lib_dirs, required: true)
  amd_lib = cc.find_library('amd', dirs: lib_dirs, required: true)
  cholmod_lib = cc.find_library('cholmod', dirs: lib_dirs, required: false)
  colamd_lib = cc.find_library('colamd', dirs: lib_dirs, required: false)
  suitesparse_lib = cc.find_library('suitesparseconfig', dirs: lib_dirs, required: false)

  umfpack_deps = [umfpack_lib, amd_lib]
  foreach lib : [cholmod_lib, colamd_lib, suitesparse_lib]
    if lib.found()
      umfpack_deps += lib
    endif
  endforeach
else
  umfpack_deps = [umfpack_dep]
endif

# Find BLAS/LAPACK
blas_dep = dependency('blas', required: false)
lapack_dep = dependency('lapack', required: false)

if not blas_dep.found()
  # Try OpenBLAS first
  openblas_lib = cc.find_library('openblas', dirs: lib_dirs, required: false)
  if openblas_lib.found()
    blas_dep = openblas_lib
    if not lapack_dep.found()
      lapack_dep = openblas_lib
    endif
  else
    blas_dep = cc.find_library('blas', dirs: lib_dirs, required: true)
  endif
endif

if not lapack_dep.found()
  lapack_dep = cc.find_library('lapack', dirs: lib_dirs, required: true)
endif

# Fortran sources (order matters for module dependencies)
fortran_src = files(
  src_dir / 'blit_const.f90',
  src_dir / 'blit_matrixutil.f90',
  src_dir / 'blit_sparseutil.f90',
  src_dir / 'blit_ilupcond.f90',
  src_dir / 'blit_blqmr.f90',
  src_dir / 'blit_blqmr_f2py.f90',
)

# C sources
c_src = files(src_dir / 'umf4_f77wrapper.c')

# f2py wrapper generation using .pyf signature file
pyf_file = files('blit_blqmr.pyf')

f2py_wrapper = custom_target('_blqmr_f2py',
  input: pyf_file,
  output: ['_blqmrmodule.c', '_blqmr-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '--build-dir', '@OUTDIR@']
)

# Platform-specific compiler flags
if is_windows
  fortran_args = ['-O3', '-cpp']
  c_args = ['-O3']
else
  fortran_args = ['-O3', '-cpp', '-fPIC']
  c_args = ['-O3', '-fPIC']
endif

# Build extension - include fortranobject.c for f2py runtime support
py.extension_module('_blqmr',
  [fortran_src, c_src, f2py_wrapper, fortranobject_c],
  include_directories: [inc_np] + suitesparse_inc_dirs,
  dependencies: [py_dep, blas_dep, lapack_dep] + umfpack_deps,
  fortran_args: fortran_args,
  c_args: c_args,
  install: true,
  subdir: 'blocksolver',
)

# Install Python files
py.install_sources(
  'blocksolver/__init__.py',
  'blocksolver/blqmr.py',
  subdir: 'blocksolver',
)
