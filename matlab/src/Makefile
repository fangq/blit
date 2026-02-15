# =============================================================================
# Makefile for BLIT BLQMR MEX extension
#
# Usage:
#   make              - build MEX file for MATLAB
#   make oct          - build .mex file for Octave
#   make bundle       - copy shared library deps for deployment (Linux)
#   make clean        - remove build artifacts
#   make test         - run MATLAB test
#   make testoct      - run Octave test
#
# Build requirements:
#   Linux:   sudo apt install libopenblas-dev libsuitesparse-dev patchelf
#   macOS:   brew install openblas suite-sparse
#   Windows: pacman -S mingw-w64-x86_64-openblas mingw-w64-x86_64-suitesparse
#
# This Makefile lives in blit/matlab/src/
# =============================================================================

# ---------- Platform detection ----------
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  PLATFORM := macos
else ifneq (,$(findstring MINGW64,$(UNAME)))
  PLATFORM := windows
else ifneq (,$(findstring MSYS,$(UNAME)))
  PLATFORM := windows
else
  PLATFORM := linux
endif

# ---------- Directory layout ----------
MKFILE_DIR  := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
MEX_DIR     := $(abspath $(MKFILE_DIR)/..)
ROOT_DIR    := $(abspath $(MEX_DIR)/..)
SRC_DIR     := $(ROOT_DIR)/src
BUILD_DIR   := $(MKFILE_DIR)/build

# ---------- Compilers ----------
FC          ?= gfortran
CC          ?= gcc
MKOCTFILE   := mkoctfile
OCTAVE      := octave

# ---------- MATLAB detection ----------
ifeq ($(PLATFORM),windows)
  MATLAB_DIR ?= $(shell where matlab 2>nul | head -1 | xargs dirname 2>/dev/null | xargs dirname 2>/dev/null)
else
  MATLAB_DIR ?= $(shell dirname $$(dirname $$(readlink -f $$(which mex))) 2>/dev/null)
endif
MATLAB_INC  := $(MATLAB_DIR)/extern/include

ifeq ($(PLATFORM),macos)
  ifeq ($(shell uname -m),arm64)
    MEX_EXT   := $(shell mexext 2>/dev/null || echo mexmaca64)
    MATLAB_LIB:= $(MATLAB_DIR)/bin/maca64
  else
    MEX_EXT   := $(shell mexext 2>/dev/null || echo mexmaci64)
    MATLAB_LIB:= $(MATLAB_DIR)/bin/maci64
  endif
else ifeq ($(PLATFORM),linux)
  MEX_EXT     := $(shell mexext 2>/dev/null || echo mexa64)
  MATLAB_LIB  := $(MATLAB_DIR)/bin/glnxa64
else
  MEX_EXT     := $(shell mexext 2>/dev/null || echo mexw64)
  MATLAB_LIB  := $(MATLAB_DIR)/bin/win64
endif

# ---------- Compiler flags ----------
ifeq ($(PLATFORM),windows)
  FFLAGS    := -O3 -cpp -fopenmp
  CFLAGS    := -O3
else
  FFLAGS    := -O3 -cpp -fPIC -fopenmp
  CFLAGS    := -O3 -fPIC
endif

# ---------- SuiteSparse detection ----------
ifeq ($(PLATFORM),macos)
  SS_INC_SEARCH := /opt/homebrew/include/suitesparse /usr/local/include/suitesparse \
                   /opt/homebrew/include /usr/local/include
  SS_LIB_SEARCH := /opt/homebrew/lib /usr/local/lib /opt/homebrew/opt/suite-sparse/lib
else ifeq ($(PLATFORM),windows)
  SS_INC_SEARCH := /mingw64/include/suitesparse /mingw64/include /ucrt64/include/suitesparse
  SS_LIB_SEARCH := /mingw64/lib /ucrt64/lib
else
  SS_INC_SEARCH := /usr/include/suitesparse /usr/local/include/suitesparse /usr/local/include
  SS_LIB_SEARCH := /usr/local/lib /usr/lib/x86_64-linux-gnu
endif

SS_INC_DIRS :=
ifdef UMFPACK_INC
  SS_INC_DIRS += $(UMFPACK_INC)
endif
SS_INC_DIRS += $(foreach d,$(SS_INC_SEARCH),$(wildcard $(d)))

SS_LIB_DIRS :=
ifdef UMFPACK_LIB
  SS_LIB_DIRS += $(UMFPACK_LIB)
endif
SS_LIB_DIRS += $(foreach d,$(SS_LIB_SEARCH),$(wildcard $(d)))

SS_INC_FLAGS := $(addprefix -I,$(SS_INC_DIRS))
SS_LIB_FLAGS := $(addprefix -L,$(SS_LIB_DIRS))

UMFPACK_LIBS := -lumfpack -lamd
_try_lib = $(if $(shell for d in $(SS_LIB_DIRS); do \
    test -f "$$d/lib$(1).a" -o -f "$$d/lib$(1).so" -o -f "$$d/lib$(1).dylib" -o -f "$$d/lib$(1).dll.a" && echo y; \
  done),-l$(1),)
UMFPACK_LIBS += $(call _try_lib,cholmod)
UMFPACK_LIBS += $(call _try_lib,colamd)
UMFPACK_LIBS += $(call _try_lib,camd)
UMFPACK_LIBS += $(call _try_lib,ccolamd)
UMFPACK_LIBS += $(call _try_lib,suitesparseconfig)
UMFPACK_LIBS += $(call _try_lib,metis)

# ---------- BLAS/LAPACK detection ----------
ifeq ($(PLATFORM),linux)
  # Prefer OpenBLAS (multi-threaded) over reference BLAS.
  # Cannot use MATLAB's -lmwblas: it uses ILP64 (64-bit ints),
  # gfortran BLAS calls use LP64 (32-bit ints) — mixing crashes.
  OPENBLAS_SO := $(firstword $(wildcard \
      /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0 \
      /usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0 \
      /usr/lib/x86_64-linux-gnu/libopenblas.so.0 \
      /usr/lib/x86_64-linux-gnu/openblas-serial/libopenblas.so.0))
  ifneq ($(OPENBLAS_SO),)
    OPENBLAS_DIR := $(dir $(OPENBLAS_SO))
    BLAS_LAPACK_LIBS = -L$(OPENBLAS_DIR) -lopenblas
  else
    BLAS_LAPACK_LIBS = -llapack -lblas
  endif

else ifeq ($(PLATFORM),macos)
  OPENBLAS_PREFIX := $(shell brew --prefix openblas 2>/dev/null)
  ifneq ($(OPENBLAS_PREFIX),)
    OPENBLAS_MACOS_A := $(wildcard $(OPENBLAS_PREFIX)/lib/libopenblas.a)
    ifneq ($(OPENBLAS_MACOS_A),)
      BLAS_LAPACK_LIBS = -force_load $(OPENBLAS_MACOS_A)
    else
      BLAS_LAPACK_LIBS = -L$(OPENBLAS_PREFIX)/lib -lopenblas
    endif
  else
    BLAS_LAPACK_LIBS = -framework Accelerate
  endif
endif

# ---------- Version script (Linux: hides internal symbols from MATLAB) ------
VERSION_MAP := $(MKFILE_DIR)/blit_mex.map

# ---------- MATLAB MEX link libraries ----------
ifeq ($(PLATFORM),linux)
  MEXLINKLIBS = -L$(MATLAB_LIB) -Wl,-rpath-link,$(MATLAB_LIB) -lmx -lmex \
                $(BLAS_LAPACK_LIBS) \
                $(SS_LIB_FLAGS) $(UMFPACK_LIBS) \
                -lgfortran -lgomp -lm -lpthread \
                -static-libgcc \
                -Wl,-Bsymbolic -Wl,--version-script,$(VERSION_MAP)

else ifeq ($(PLATFORM),macos)
  MEXLINKLIBS = -L$(MATLAB_LIB) -Wl,-rpath,$(MATLAB_LIB) -lmx -lmex \
                $(BLAS_LAPACK_LIBS) \
                $(SS_LIB_FLAGS) $(UMFPACK_LIBS) \
                -lgfortran -lgomp -lm \
                -Wl,-rpath,@loader_path/blqmr_deps \
                -Wl,-exported_symbol,_mexFunction \
                -Wl,-exported_symbol,_mexfunction_

else
  MEXLINKLIBS = -L$(MATLAB_LIB) -lmx -lmex \
                -lopenblas \
                $(SS_LIB_FLAGS) $(UMFPACK_LIBS) \
                -lgfortran -lgomp -lm \
                -static-libgcc -static-libgfortran \
                -Wl,-Bstatic -lgomp -lwinpthread -Wl,-Bdynamic
endif

# ---------- Octave link libraries ----------
ifeq ($(PLATFORM),linux)
  OCT_LINKLIBS = $(SS_LIB_FLAGS) $(UMFPACK_LIBS) $(BLAS_LAPACK_LIBS) \
                 -lgfortran -lgomp -lm -static-libgcc
else
  OCT_LINKLIBS = $(SS_LIB_FLAGS) $(UMFPACK_LIBS) -llapack -lblas \
                 -lgfortran -lgomp -lm
endif

# ---------- Source files ----------
F_SRCS := $(SRC_DIR)/blit_const.f90 \
          $(SRC_DIR)/blit_matrixutil.f90 \
          $(SRC_DIR)/blit_sparseutil.f90 \
          $(SRC_DIR)/blit_ilupcond.f90 \
          $(SRC_DIR)/blit_blqmr.f90 \
          $(SRC_DIR)/blit_blqmr_f2py.f90

C_SRCS := $(SRC_DIR)/umf4_f77wrapper.c \
          $(SRC_DIR)/blit_blas_threads.c

# ---------- MATLAB MEX objects ----------
MEX_GATE     := $(MKFILE_DIR)/blqmr_mex.f90
MEX_GATE_OBJ := $(BUILD_DIR)/blqmr_mex.o
F_OBJS := $(patsubst $(SRC_DIR)/%.f90,$(BUILD_DIR)/%.o,$(F_SRCS))
C_OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SRCS))
ALL_OBJS := $(F_OBJS) $(C_OBJS)
TARGET := $(MEX_DIR)/blqmr_.$(MEX_EXT)

# ---------- Octave MEX objects ----------
BUILD_DIR_OCT := $(BUILD_DIR)/oct
OCT_MEX_GATE  := $(MKFILE_DIR)/blqmr_mex.c
OCT_GATE_OBJ  := $(BUILD_DIR_OCT)/blqmr_mex.o
OCT_F_OBJS    := $(patsubst $(SRC_DIR)/%.f90,$(BUILD_DIR_OCT)/%.o,$(F_SRCS))
OCT_C_OBJS    := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR_OCT)/%.o,$(C_SRCS))
OCT_ALL_OBJS  := $(OCT_F_OBJS) $(OCT_C_OBJS)
OCT_TARGET    := $(MEX_DIR)/blqmr_.mex

# ---------- Bundle paths ----------
BUNDLE_DIR    := $(MEX_DIR)/blqmr_deps
BUNDLE_SCRIPT := $(MKFILE_DIR)/bundle_deps.sh

# =============================================================================
# Top-level targets
# =============================================================================

.PHONY: all oct clean bundle test testoct dirs

all: dirs $(TARGET)

oct: dirs $(OCT_TARGET)

dirs:
	@mkdir -p $(BUILD_DIR) $(BUILD_DIR_OCT)

# =============================================================================
# MATLAB MEX build
# =============================================================================

$(MEX_GATE_OBJ): $(MEX_GATE) $(F_OBJS) | dirs
	$(FC) $(FFLAGS) -DMATLAB_DEFAULT_RELEASE=R2017b -I$(MATLAB_INC) -I$(BUILD_DIR) -c $< -o $@

ifeq ($(PLATFORM),windows)
$(TARGET): $(MEX_GATE_OBJ) $(ALL_OBJS)
	$(FC) -shared -o $@ $(MEX_GATE_OBJ) $(ALL_OBJS) $(MEXLINKLIBS) \
	  -Wl,--output-def,$(BUILD_DIR)/blqmr_.def
else ifeq ($(PLATFORM),linux)
$(TARGET): $(MEX_GATE_OBJ) $(ALL_OBJS)
	$(FC) -shared -o $@ $(MEX_GATE_OBJ) $(ALL_OBJS) $(MEXLINKLIBS)
	patchelf --set-rpath '$$ORIGIN/blqmr_deps' $@
else
$(TARGET): $(MEX_GATE_OBJ) $(ALL_OBJS)
	$(FC) -shared -o $@ $(MEX_GATE_OBJ) $(ALL_OBJS) $(MEXLINKLIBS)
endif

# ---------- MATLAB Fortran objects ----------
$(BUILD_DIR)/blit_const.o: $(SRC_DIR)/blit_const.f90 | dirs
	$(FC) $(FFLAGS) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/blit_matrixutil.o: $(SRC_DIR)/blit_matrixutil.f90 $(BUILD_DIR)/blit_const.o
	$(FC) $(FFLAGS) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/blit_sparseutil.o: $(SRC_DIR)/blit_sparseutil.f90 $(BUILD_DIR)/blit_const.o
	$(FC) $(FFLAGS) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/blit_ilupcond.o: $(SRC_DIR)/blit_ilupcond.f90 $(BUILD_DIR)/blit_const.o
	$(FC) $(FFLAGS) $(SS_INC_FLAGS) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/blit_blqmr.o: $(SRC_DIR)/blit_blqmr.f90 \
    $(BUILD_DIR)/blit_const.o $(BUILD_DIR)/blit_ilupcond.o \
    $(BUILD_DIR)/blit_sparseutil.o $(BUILD_DIR)/blit_matrixutil.o
	$(FC) $(FFLAGS) -I$(BUILD_DIR) -I$(SRC_DIR) -J$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/blit_blqmr_f2py.o: $(SRC_DIR)/blit_blqmr_f2py.f90 \
    $(BUILD_DIR)/blit_blqmr.o $(BUILD_DIR)/blit_const.o
	$(FC) $(FFLAGS) -I$(BUILD_DIR) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | dirs
	$(CC) $(CFLAGS) $(SS_INC_FLAGS) -c $< -o $@

# =============================================================================
# Octave MEX build
# =============================================================================

$(OCT_GATE_OBJ): $(OCT_MEX_GATE) $(OCT_F_OBJS) | dirs
	$(MKOCTFILE) --mex -c $(SS_INC_FLAGS) $< -o $@

ifeq ($(PLATFORM),linux)
$(OCT_TARGET): $(OCT_GATE_OBJ) $(OCT_ALL_OBJS)
	$(MKOCTFILE) --mex $(OCT_GATE_OBJ) $(OCT_ALL_OBJS) $(OCT_LINKLIBS) -o $@
	patchelf --set-rpath '$$ORIGIN/blqmr_deps' $@
else
$(OCT_TARGET): $(OCT_GATE_OBJ) $(OCT_ALL_OBJS)
	$(MKOCTFILE) --mex $(OCT_GATE_OBJ) $(OCT_ALL_OBJS) $(OCT_LINKLIBS) -o $@
endif

# ---------- Octave Fortran objects ----------
$(BUILD_DIR_OCT)/blit_const.o: $(SRC_DIR)/blit_const.f90 | dirs
	$(FC) $(FFLAGS) -J$(BUILD_DIR_OCT) -c $< -o $@

$(BUILD_DIR_OCT)/blit_matrixutil.o: $(SRC_DIR)/blit_matrixutil.f90 $(BUILD_DIR_OCT)/blit_const.o
	$(FC) $(FFLAGS) -I$(BUILD_DIR_OCT) -J$(BUILD_DIR_OCT) -c $< -o $@

$(BUILD_DIR_OCT)/blit_sparseutil.o: $(SRC_DIR)/blit_sparseutil.f90 $(BUILD_DIR_OCT)/blit_const.o
	$(FC) $(FFLAGS) -I$(BUILD_DIR_OCT) -J$(BUILD_DIR_OCT) -c $< -o $@

$(BUILD_DIR_OCT)/blit_ilupcond.o: $(SRC_DIR)/blit_ilupcond.f90 $(BUILD_DIR_OCT)/blit_const.o
	$(FC) $(FFLAGS) $(SS_INC_FLAGS) -I$(BUILD_DIR_OCT) -J$(BUILD_DIR_OCT) -c $< -o $@

$(BUILD_DIR_OCT)/blit_blqmr.o: $(SRC_DIR)/blit_blqmr.f90 \
    $(BUILD_DIR_OCT)/blit_const.o $(BUILD_DIR_OCT)/blit_ilupcond.o \
    $(BUILD_DIR_OCT)/blit_sparseutil.o $(BUILD_DIR_OCT)/blit_matrixutil.o
	$(FC) $(FFLAGS) -I$(BUILD_DIR_OCT) -I$(SRC_DIR) -J$(BUILD_DIR_OCT) -c $< -o $@

$(BUILD_DIR_OCT)/blit_blqmr_f2py.o: $(SRC_DIR)/blit_blqmr_f2py.f90 \
    $(BUILD_DIR_OCT)/blit_blqmr.o $(BUILD_DIR_OCT)/blit_const.o
	$(FC) $(FFLAGS) -I$(BUILD_DIR_OCT) -c $< -o $@

$(BUILD_DIR_OCT)/%.o: $(SRC_DIR)/%.c | dirs
	$(CC) $(CFLAGS) $(SS_INC_FLAGS) -c $< -o $@

# =============================================================================
# Bundle shared library dependencies for deployment
# =============================================================================
#
# Collects non-system shared library deps into blqmr_deps/ alongside the MEX.
# The MEX rpath ($ORIGIN/blqmr_deps, set by patchelf during build) ensures
# bundled libs are found first at runtime.
#
# Requires: bundle_deps.sh (in this directory), patchelf (Linux)
# To deploy: ship blqmr_.mexa64 + blqmr_deps/ + .m files together.

bundle: $(TARGET)
	@echo "=== Bundling shared library dependencies ==="
	@rm -rf $(BUNDLE_DIR)
	@sh $(BUNDLE_SCRIPT) $(TARGET) $(BUNDLE_DIR)
	@if [ -f $(OCT_TARGET) ]; then \
	    echo "Also bundling Octave MEX dependencies..."; \
	    sh $(BUNDLE_SCRIPT) $(OCT_TARGET) $(BUNDLE_DIR); \
	fi
	@echo ""
	@echo "=== Bundle complete ==="
	@echo "Deploy these together:"
	@echo "  $(TARGET)"
	@test -f $(OCT_TARGET) && echo "  $(OCT_TARGET)" || true
	@echo "  $(BUNDLE_DIR)/"
	@ls -lh $(BUNDLE_DIR)/
	@echo ""
	@echo "Verify: LD_LIBRARY_PATH=$(BUNDLE_DIR) ldd $(TARGET)"

# =============================================================================
# Housekeeping
# =============================================================================

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(MEX_DIR)/blqmr_.mex*
	rm -rf $(BUNDLE_DIR)

test:
	matlab -batch "addpath('$(MEX_DIR)'); blqmr_compare"

testoct:
	$(OCTAVE) --eval "addpath('$(MEX_DIR)'); blqmr_compare"