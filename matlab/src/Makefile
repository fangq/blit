# =============================================================================
# Makefile for BLIT BLQMR MEX extension
#
# Usage:
#   make              - build MEX file
#   make clean        - remove build artifacts
#   make test         - run MATLAB test
#
# Supported platforms: Linux, macOS, Windows (MSYS2/MinGW64)
#
# This Makefile lives in blit/matlab/src/
# =============================================================================

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  PLATFORM := macos
else ifneq (,$(findstring MINGW64,$(UNAME)))
  PLATFORM := windows
else ifneq (,$(findstring MSYS,$(UNAME)))
  PLATFORM := windows
else ifneq (,$(findstring CYGWIN,$(UNAME)))
  PLATFORM := windows
else ifeq ($(UNAME),Linux)
  PLATFORM := linux
else
  PLATFORM := linux
endif

MKFILE_DIR  := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
MEX_DIR     := $(abspath $(MKFILE_DIR)/..)
ROOT_DIR    := $(abspath $(MEX_DIR)/..)
SRC_DIR     := $(ROOT_DIR)/src
PRIVATE_DIR := $(MEX_DIR)/private
BUILD_DIR   := $(MKFILE_DIR)/build

FC          ?= gfortran
CC          ?= gcc

# ---------- MATLAB detection ----------
ifeq ($(PLATFORM),windows)
  MATLAB_DIR  ?= $(shell where matlab 2>nul | head -1 | xargs dirname 2>/dev/null | xargs dirname 2>/dev/null)
  ifeq ($(MATLAB_DIR),)
    MATLAB_DIR ?= C:/Program Files/MATLAB/R2024b
  endif
else
  MATLAB_DIR  ?= $(shell dirname $$(dirname $$(readlink -f $$(which mex))) 2>/dev/null)
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
  MEX_EXT   := $(shell mexext 2>/dev/null || echo mexa64)
  MATLAB_LIB:= $(MATLAB_DIR)/bin/glnxa64
else
  MEX_EXT   := $(shell mexext 2>/dev/null || echo mexw64)
  MATLAB_LIB:= $(MATLAB_DIR)/bin/win64
endif

# ---------- Compiler flags ----------
FFLAGS_BASE := -O3 -cpp -fPIC -fopenmp
CFLAGS_BASE := -O3 -fPIC

ifeq ($(PLATFORM),windows)
  # MinGW: no -fPIC needed
  FFLAGS    := -O3 -cpp -fopenmp
  CFLAGS    := -O3
else
  FFLAGS    := $(FFLAGS_BASE)
  CFLAGS    := $(CFLAGS_BASE)
endif

# ---------- SuiteSparse ----------
ifeq ($(PLATFORM),macos)
  SS_INC_SEARCH := /opt/homebrew/include/suitesparse \
                   /usr/local/include/suitesparse \
                   /opt/homebrew/include \
                   /usr/local/include
  SS_LIB_SEARCH := /opt/homebrew/lib \
                   /usr/local/lib \
                   /opt/homebrew/opt/suite-sparse/lib
else ifeq ($(PLATFORM),windows)
  SS_INC_SEARCH := /mingw64/include/suitesparse \
                   /mingw64/include \
                   /ucrt64/include/suitesparse \
                   /ucrt64/include
  SS_LIB_SEARCH := /mingw64/lib \
                   /ucrt64/lib
else
  SS_INC_SEARCH := /usr/include/suitesparse \
                   /usr/local/include/suitesparse \
                   /usr/local/include
  SS_LIB_SEARCH := /usr/local/lib \
                   /usr/lib/x86_64-linux-gnu
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

# ---------- Version script (Linux/macOS: hide BLAS from MKL interposition) ----------
VERSION_MAP := $(MKFILE_DIR)/blit_mex.map

# ---------- Static BLAS/LAPACK ----------
# MATLAB's MKL intercepts dgemm_ at runtime, causing crashes.
# Statically link BLAS so dgemm_ is defined inside the .mexa64/.mexmaca64,
# then hidden by version script / -Bsymbolic.

ifeq ($(PLATFORM),linux)
  NETLIB_BLAS   := $(firstword $(wildcard \
      /usr/lib/x86_64-linux-gnu/blas/libblas.a \
      /usr/lib/blas/libblas.a))
  NETLIB_LAPACK := $(firstword $(wildcard \
      /usr/lib/x86_64-linux-gnu/lapack/liblapack.a \
      /usr/lib/lapack/liblapack.a))
  OPENBLAS_A    := $(firstword $(wildcard \
      /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.a \
      /usr/lib/x86_64-linux-gnu/openblas-serial/libblas.a \
      /usr/lib/x86_64-linux-gnu/libblas.a))

  ifneq ($(NETLIB_BLAS),)
    # Netlib: link on demand (no --whole-archive to avoid non-PIC cblas objects).
    # Order: LAPACK first, then BLAS (LAPACK depends on BLAS).
    # --allow-multiple-definition handles lsame_/xerbla_ duplicates.
    STATIC_BLAS_LAPACK = -Wl,--allow-multiple-definition \
        $(NETLIB_LAPACK) $(NETLIB_BLAS)
  else ifneq ($(OPENBLAS_A),)
    STATIC_BLAS_LAPACK = \
        -Wl,--whole-archive $(OPENBLAS_A) -Wl,--no-whole-archive
  else
    $(error No static libblas.a found. Install libblas-dev or libopenblas-dev)
  endif

  MEXLINKLIBS = -L$(MATLAB_LIB) -Wl,-rpath-link,$(MATLAB_LIB) -lmx -lmex \
                $(SS_LIB_FLAGS) $(UMFPACK_LIBS) \
                $(STATIC_BLAS_LAPACK) \
                -lgfortran -lgomp -lm \
                -Wl,-Bsymbolic -Wl,--version-script,$(VERSION_MAP)

else ifeq ($(PLATFORM),macos)
  # macOS: use Accelerate framework or Homebrew OpenBLAS
  OPENBLAS_PREFIX := $(shell brew --prefix openblas 2>/dev/null)
  ifneq ($(OPENBLAS_PREFIX),)
    # Homebrew OpenBLAS - static .a is built with -fPIC on macOS
    OPENBLAS_MACOS_A := $(wildcard $(OPENBLAS_PREFIX)/lib/libopenblas.a)
    ifneq ($(OPENBLAS_MACOS_A),)
      STATIC_BLAS_LAPACK = -force_load $(OPENBLAS_MACOS_A)
    else
      STATIC_BLAS_LAPACK = -L$(OPENBLAS_PREFIX)/lib -lopenblas
    endif
  else
    # Fallback: Apple Accelerate (no MKL conflict on macOS)
    STATIC_BLAS_LAPACK = -framework Accelerate
  endif

  # macOS uses -exported_symbols_list instead of --version-script
  MEXLINKLIBS = -L$(MATLAB_LIB) -Wl,-rpath,$(MATLAB_LIB) -lmx -lmex \
                $(SS_LIB_FLAGS) $(UMFPACK_LIBS) \
                $(STATIC_BLAS_LAPACK) \
                -lgfortran -lgomp -lm \
                -Wl,-exported_symbol,_mexFunction \
                -Wl,-exported_symbol,_mexfunction_

else
  # Windows (MSYS2/MinGW): static link everything
  MEXLINKLIBS = -L$(MATLAB_LIB) -lmx -lmex \
                $(SS_LIB_FLAGS) $(UMFPACK_LIBS) \
                -lopenblas -lgfortran -lgomp -lm \
                -static-libgcc -static-libgfortran \
                -Wl,-Bstatic -lgomp -lwinpthread -Wl,-Bdynamic
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

MEX_GATE     := $(MKFILE_DIR)/blqmr_mex.f90
MEX_GATE_OBJ := $(BUILD_DIR)/blqmr_mex.o

F_OBJS := $(patsubst $(SRC_DIR)/%.f90,$(BUILD_DIR)/%.o,$(F_SRCS))
C_OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SRCS))
ALL_OBJS := $(F_OBJS) $(C_OBJS)

TARGET := $(MEX_DIR)/blqmr_.$(MEX_EXT)

# =============================================================================
# Rules
# =============================================================================

.PHONY: all clean test dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BUILD_DIR)

# ---------- Compile Fortran MEX gateway ----------
$(MEX_GATE_OBJ): $(MEX_GATE) $(F_OBJS) | dirs
	$(FC) $(FFLAGS) -DMATLAB_DEFAULT_RELEASE=R2017b -I$(MATLAB_INC) -I$(BUILD_DIR) -c $< -o $@

# ---------- Link ----------
ifeq ($(PLATFORM),windows)
# Windows: use gfortran -shared, output .mexw64
$(TARGET): $(MEX_GATE_OBJ) $(ALL_OBJS)
	$(FC) -shared -o $@ $(MEX_GATE_OBJ) $(ALL_OBJS) $(MEXLINKLIBS) \
	  -Wl,--output-def,$(BUILD_DIR)/blqmr_.def
else
# Linux/macOS: use gfortran -shared
$(TARGET): $(MEX_GATE_OBJ) $(ALL_OBJS)
	$(FC) -shared -o $@ $(MEX_GATE_OBJ) $(ALL_OBJS) $(MEXLINKLIBS)
endif

# ---------- Fortran objects ----------
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

$(BUILD_DIR)/blit_blqmr_f2py.o: $(SRC_DIR)/blit_blqmr_f2py.f90 $(BUILD_DIR)/blit_blqmr.o $(BUILD_DIR)/blit_const.o
	$(FC) $(FFLAGS) -I$(BUILD_DIR) -c $< -o $@

# ---------- C objects ----------
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | dirs
	$(CC) $(CFLAGS) $(SS_INC_FLAGS) -c $< -o $@

# ---------- Housekeeping ----------
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(MEX_DIR)/blqmr_.mex*

test:
	matlab -batch "addpath('$(MEX_DIR)'); blqmr_test"