# Meson build for BLIT Python extension
#
# Run from python/ directory:
#   pip install . 
#   # or
#   pip install . --no-build-isolation
#   # or for development
#   pip install -e . --no-build-isolation

project('blocksolver', ['c', 'fortran'],
  version: '0.9.0',
  license: 'BSD-3-Clause OR LGPL-3.0-or-later',
  meson_version: '>=0.64.0',
  default_options: ['buildtype=release', 'default_library=static', 'prefer_static=true']
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

# Find UMFPACK and dependencies - prefer static libraries
umfpack_deps = []

# Try to find static libraries first, fall back to shared
# Use static: true with required: true to enforce static linking where possible
umfpack_lib = cc.find_library('umfpack', dirs: lib_dirs, static: true, required: false)
if not umfpack_lib.found()
  umfpack_lib = cc.find_library('umfpack', dirs: lib_dirs, required: true)
  message('WARNING: Using shared libumfpack - wheel may need auditwheel')
endif
umfpack_deps += umfpack_lib

amd_lib = cc.find_library('amd', dirs: lib_dirs, static: true, required: false)
if not amd_lib.found()
  amd_lib = cc.find_library('amd', dirs: lib_dirs, required: true)
  message('WARNING: Using shared libamd - wheel may need auditwheel')
endif
umfpack_deps += amd_lib

# Optional SuiteSparse dependencies
cholmod_lib = cc.find_library('cholmod', dirs: lib_dirs, static: true, required: false)
if not cholmod_lib.found()
  cholmod_lib = cc.find_library('cholmod', dirs: lib_dirs, required: false)
endif
if cholmod_lib.found()
  umfpack_deps += cholmod_lib
endif

colamd_lib = cc.find_library('colamd', dirs: lib_dirs, static: true, required: false)
if not colamd_lib.found()
  colamd_lib = cc.find_library('colamd', dirs: lib_dirs, required: false)
endif
if colamd_lib.found()
  umfpack_deps += colamd_lib
endif

camd_lib = cc.find_library('camd', dirs: lib_dirs, static: true, required: false)
if not camd_lib.found()
  camd_lib = cc.find_library('camd', dirs: lib_dirs, required: false)
endif
if camd_lib.found()
  umfpack_deps += camd_lib
endif

ccolamd_lib = cc.find_library('ccolamd', dirs: lib_dirs, static: true, required: false)
if not ccolamd_lib.found()
  ccolamd_lib = cc.find_library('ccolamd', dirs: lib_dirs, required: false)
endif
if ccolamd_lib.found()
  umfpack_deps += ccolamd_lib
endif

suitesparse_lib = cc.find_library('suitesparseconfig', dirs: lib_dirs, static: true, required: false)
if not suitesparse_lib.found()
  suitesparse_lib = cc.find_library('suitesparseconfig', dirs: lib_dirs, required: false)
endif
if suitesparse_lib.found()
  umfpack_deps += suitesparse_lib
endif

# Find BLAS/LAPACK - prefer static OpenBLAS which includes both
blas_lapack_deps = []

openblas_lib = cc.find_library('openblas', dirs: lib_dirs, static: true, required: false)
if not openblas_lib.found()
  openblas_lib = cc.find_library('openblas', dirs: lib_dirs, required: false)
endif

if openblas_lib.found()
  blas_lapack_deps += openblas_lib
else
  # Fall back to separate BLAS and LAPACK
  blas_lib = cc.find_library('blas', dirs: lib_dirs, static: true, required: false)
  if not blas_lib.found()
    blas_lib = cc.find_library('blas', dirs: lib_dirs, required: true)
  endif
  blas_lapack_deps += blas_lib
  
  lapack_lib = cc.find_library('lapack', dirs: lib_dirs, static: true, required: false)
  if not lapack_lib.found()
    lapack_lib = cc.find_library('lapack', dirs: lib_dirs, required: true)
  endif
  blas_lapack_deps += lapack_lib
endif

# OpenMP dependency (needed by OpenBLAS and SuiteSparse when built with OpenMP)
omp_dep = dependency('openmp', required: false)
if not omp_dep.found()
  # Try to find libgomp directly
  omp_lib = cc.find_library('gomp', dirs: lib_dirs, static: true, required: false)
  if not omp_lib.found()
    omp_lib = cc.find_library('gomp', dirs: lib_dirs, required: false)
  endif
  if not omp_lib.found()
    omp_lib = cc.find_library('omp', dirs: lib_dirs, required: false)
  endif
  if omp_lib.found()
    omp_dep = omp_lib
  endif
endif

# On Windows with MinGW, we may need additional libraries for static linking
if is_windows
  # GFortran runtime (usually needed for static Fortran libs)
  gfortran_lib = cc.find_library('gfortran', dirs: lib_dirs, static: true, required: false)
  if gfortran_lib.found()
    blas_lapack_deps += gfortran_lib
  endif
  
  # QuadMath (needed by gfortran)
  quadmath_lib = cc.find_library('quadmath', dirs: lib_dirs, static: true, required: false)
  if quadmath_lib.found()
    blas_lapack_deps += quadmath_lib
  endif
  
  # Windows threading
  pthread_lib = cc.find_library('pthread', dirs: lib_dirs, static: true, required: false)
  if not pthread_lib.found()
    pthread_lib = cc.find_library('winpthread', dirs: lib_dirs, static: true, required: false)
  endif
  if pthread_lib.found()
    blas_lapack_deps += pthread_lib
  endif
endif

# Collect all dependencies
all_deps = [py_dep] + umfpack_deps + blas_lapack_deps
if omp_dep.found()
  all_deps += omp_dep
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
c_src = files(src_dir / 'umf4_f77wrapper.c', src_dir / 'blit_blas_threads.c')

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
  # Static linking flags for Windows
  link_args = ['-static-libgcc', '-static-libgfortran']
elif is_macos
  fortran_args = ['-O3', '-cpp', '-fPIC', '-fopenmp']
  c_args = ['-O3', '-fPIC']
    # Static link libgomp: find libgomp.a from gfortran
  gomp_static = fc.find_library('gomp', static: true, required: false)
  if gomp_static.found()
    link_args = ['-Wl,-Bstatic', '-lgomp', '-Wl,-Bdynamic', '-lpthread']
  else
    link_args = ['-fopenmp']
  endif
else
  # Linux
  fortran_args = ['-O3', '-cpp', '-fPIC', '-fopenmp']
  c_args = ['-O3', '-fPIC']
  # Static link libgomp on Linux: -Bstatic for gomp, -Bdynamic for pthread
  link_args = ['-static-libgcc', '-static-libgfortran',
               '-Wl,-Bstatic', '-lgomp', '-Wl,-Bdynamic', '-lpthread']
endif

# Build extension - include fortranobject.c for f2py runtime support
py.extension_module('_blqmr',
  [fortran_src, c_src, f2py_wrapper, fortranobject_c],
  include_directories: [inc_np] + suitesparse_inc_dirs,
  dependencies: all_deps,
  fortran_args: fortran_args,
  c_args: c_args,
  link_args: link_args,
  install: true,
  subdir: 'blocksolver',
)

# Install Python files
py.install_sources(
  'blocksolver/__init__.py',
  'blocksolver/blqmr.py',
  subdir: 'blocksolver',
)