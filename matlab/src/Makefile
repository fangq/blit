# =============================================================================
# Makefile for BLIT BLQMR MEX extension
#
# Usage:
#   make              - build MEX file for MATLAB
#   make oct          - build .mex file for Octave
#   make clean        - remove build artifacts
#   make test         - run MATLAB test
#   make testoct      - run Octave test
#
# Supported platforms: Linux, macOS, Windows (MSYS2/MinGW64)
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
PRIVATE_DIR := $(MEX_DIR)/private
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

# ---------- SuiteSparse ----------
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

# ---------- Version script (hides BLAS symbols from MATLAB's MKL) ----------
VERSION_MAP := $(MKFILE_DIR)/blit_mex.map

# ---------- Static BLAS/LAPACK for MATLAB MEX ----------
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
    STATIC_BLAS_LAPACK = -Wl,--allow-multiple-definition $(NETLIB_LAPACK) $(NETLIB_BLAS)
  else ifneq ($(OPENBLAS_A),)
    STATIC_BLAS_LAPACK = -Wl,--whole-archive $(OPENBLAS_A) -Wl,--no-whole-archive
  else
    $(error No static libblas.a found. Install libblas-dev or libopenblas-dev)
  endif

  MEXLINKLIBS = -L$(MATLAB_LIB) -Wl,-rpath-link,$(MATLAB_LIB) -lmx -lmex \
                $(SS_LIB_FLAGS) $(UMFPACK_LIBS) \
                $(STATIC_BLAS_LAPACK) \
                -lgfortran -lgomp -lm \
                -Wl,-Bsymbolic -Wl,--version-script,$(VERSION_MAP)

else ifeq ($(PLATFORM),macos)
  OPENBLAS_PREFIX := $(shell brew --prefix openblas 2>/dev/null)
  ifneq ($(OPENBLAS_PREFIX),)
    OPENBLAS_MACOS_A := $(wildcard $(OPENBLAS_PREFIX)/lib/libopenblas.a)
    ifneq ($(OPENBLAS_MACOS_A),)
      STATIC_BLAS_LAPACK = -force_load $(OPENBLAS_MACOS_A)
    else
      STATIC_BLAS_LAPACK = -L$(OPENBLAS_PREFIX)/lib -lopenblas
    endif
  else
    STATIC_BLAS_LAPACK = -framework Accelerate
  endif

  MEXLINKLIBS = -L$(MATLAB_LIB) -Wl,-rpath,$(MATLAB_LIB) -lmx -lmex \
                $(SS_LIB_FLAGS) $(UMFPACK_LIBS) \
                $(STATIC_BLAS_LAPACK) \
                -lgfortran -lgomp -lm \
                -Wl,-exported_symbol,_mexFunction \
                -Wl,-exported_symbol,_mexfunction_

else
  MEXLINKLIBS = -L$(MATLAB_LIB) -lmx -lmex \
                $(SS_LIB_FLAGS) $(UMFPACK_LIBS) \
                -lopenblas -lgfortran -lgomp -lm \
                -static-libgcc -static-libgfortran \
                -Wl,-Bstatic -lgomp -lwinpthread -Wl,-Bdynamic
endif

# ---------- Octave link libraries ----------
# Use static Netlib BLAS for Octave too (avoids OpenBLAS complex issues)
ifeq ($(PLATFORM),linux)
  ifneq ($(NETLIB_BLAS),)
    OCT_LINKLIBS = $(SS_LIB_FLAGS) $(UMFPACK_LIBS) \
                   -Wl,--allow-multiple-definition $(NETLIB_LAPACK) $(NETLIB_BLAS) \
                   -lgfortran -lgomp -lm
  else
    OCT_LINKLIBS = $(SS_LIB_FLAGS) $(UMFPACK_LIBS) -llapack -lblas -lgfortran -lgomp -lm
  endif
else
  OCT_LINKLIBS = $(SS_LIB_FLAGS) $(UMFPACK_LIBS) -llapack -lblas -lgfortran -lgomp -lm
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

# ---------- MATLAB MEX files ----------
MEX_GATE     := $(MKFILE_DIR)/blqmr_mex.f90
MEX_GATE_OBJ := $(BUILD_DIR)/blqmr_mex.o

F_OBJS := $(patsubst $(SRC_DIR)/%.f90,$(BUILD_DIR)/%.o,$(F_SRCS))
C_OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SRCS))
ALL_OBJS := $(F_OBJS) $(C_OBJS)

TARGET := $(MEX_DIR)/blqmr_.$(MEX_EXT)

# ---------- Octave MEX files ----------
BUILD_DIR_OCT := $(BUILD_DIR)/oct
OCT_MEX_GATE  := $(MKFILE_DIR)/blqmr_mex.c
OCT_GATE_OBJ  := $(BUILD_DIR_OCT)/blqmr_mex.o
OCT_F_OBJS    := $(patsubst $(SRC_DIR)/%.f90,$(BUILD_DIR_OCT)/%.o,$(F_SRCS))
OCT_C_OBJS    := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR_OCT)/%.o,$(C_SRCS))
OCT_ALL_OBJS  := $(OCT_F_OBJS) $(OCT_C_OBJS)
OCT_TARGET    := $(MEX_DIR)/blqmr_.mex

# =============================================================================
# Rules
# =============================================================================

.PHONY: all oct clean test testoct dirs

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

$(OCT_TARGET): $(OCT_GATE_OBJ) $(OCT_ALL_OBJS)
	$(MKOCTFILE) --mex $(OCT_GATE_OBJ) $(OCT_ALL_OBJS) $(OCT_LINKLIBS) -o $@

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
# Housekeeping
# =============================================================================

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(MEX_DIR)/blqmr_.mex*

test:
	matlab -batch "addpath('$(MEX_DIR)'); blqmr_compare"

testoct:
	$(OCTAVE) --eval "addpath('$(MEX_DIR)'); blqmr_compare"