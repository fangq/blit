# =============================================================================
# Makefile for BLIT BLQMR MEX extension
#
# Usage:
#   make              - build MEX file
#   make clean        - remove build artifacts
#   make test         - run MATLAB test
#
# This Makefile lives in blit/matlab/src/
# =============================================================================

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  PLATFORM := macos
else ifeq ($(UNAME),Linux)
  PLATFORM := linux
else
  PLATFORM := windows
endif

MKFILE_DIR  := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
MEX_DIR     := $(abspath $(MKFILE_DIR)/..)
ROOT_DIR    := $(abspath $(MEX_DIR)/..)
SRC_DIR     := $(ROOT_DIR)/src
PRIVATE_DIR := $(MEX_DIR)/private
BUILD_DIR   := $(MKFILE_DIR)/build

FC          ?= gfortran
CC          ?= gcc

MATLAB_DIR  ?= $(shell dirname $$(dirname $$(readlink -f $$(which mex))) 2>/dev/null)
MATLAB_INC  := $(MATLAB_DIR)/extern/include

ifeq ($(PLATFORM),macos)
  MEX_EXT   := $(shell mexext 2>/dev/null || echo mexmaca64)
  MATLAB_LIB:= $(MATLAB_DIR)/bin/maca64
else ifeq ($(PLATFORM),linux)
  MEX_EXT   := $(shell mexext 2>/dev/null || echo mexa64)
  MATLAB_LIB:= $(MATLAB_DIR)/bin/glnxa64
else
  MEX_EXT   := $(shell mexext 2>/dev/null || echo mexw64)
  MATLAB_LIB:= $(MATLAB_DIR)/bin/win64
endif

FFLAGS      := -O3 -cpp -fPIC -fopenmp
CFLAGS      := -O3 -fPIC

# ---------- SuiteSparse ----------
SS_INC_SEARCH := /opt/homebrew/include/suitesparse \
                 /usr/local/include/suitesparse \
                 /usr/include/suitesparse \
                 /opt/homebrew/include \
                 /usr/local/include

SS_LIB_SEARCH := /opt/homebrew/lib \
                 /usr/local/lib \
                 /usr/lib/x86_64-linux-gnu \
                 /opt/homebrew/opt/suite-sparse/lib

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
    test -f "$$d/lib$(1).a" -o -f "$$d/lib$(1).so" -o -f "$$d/lib$(1).dylib" && echo y; \
  done),-l$(1),)

UMFPACK_LIBS += $(call _try_lib,cholmod)
UMFPACK_LIBS += $(call _try_lib,colamd)
UMFPACK_LIBS += $(call _try_lib,camd)
UMFPACK_LIBS += $(call _try_lib,ccolamd)
UMFPACK_LIBS += $(call _try_lib,suitesparseconfig)
UMFPACK_LIBS += $(call _try_lib,metis)

# ---------- Link libraries ----------
VERSION_MAP := $(MKFILE_DIR)/blit_mex.map

# ---------- Find static BLAS/LAPACK ----------
# Need BLAS statically linked so dgemm_ is defined inside the .mexa64,
# hidden by version script to prevent MATLAB's MKL from interposing.
#
# Priority: Netlib reference (separate blas/lapack dirs), then OpenBLAS.
# OpenBLAS bundles LAPACK inside libblas.a, so only one archive is needed.
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
  # Netlib: separate BLAS + LAPACK archives
  STATIC_BLAS_LAPACK = $(NETLIB_LAPACK) -Wl,--whole-archive $(NETLIB_BLAS) -Wl,--no-whole-archive
else ifneq ($(OPENBLAS_A),)
  # OpenBLAS: single archive with BLAS+LAPACK bundled
  STATIC_BLAS_LAPACK = -Wl,--whole-archive $(OPENBLAS_A) -Wl,--no-whole-archive
else
  $(error No static libblas.a found. Install libblas-dev or libopenblas-dev)
endif

# ---------- Link libraries ----------
# Static BLAS/LAPACK + version script + -Bsymbolic = dgemm_ hidden from MKL.
MEXLINKLIBS = -L$(MATLAB_LIB) -Wl,-rpath-link,$(MATLAB_LIB) -lmx -lmex \
              $(SS_LIB_FLAGS) $(UMFPACK_LIBS) \
              $(STATIC_BLAS_LAPACK) \
              -lgfortran -lgomp -lm \
              -Wl,-Bsymbolic -Wl,--version-script,$(VERSION_MAP)

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

TARGET := $(PRIVATE_DIR)/blqmr_.$(MEX_EXT)

# =============================================================================

.PHONY: all clean test dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BUILD_DIR) $(PRIVATE_DIR)

# ---------- Compile Fortran MEX gateway ----------
$(MEX_GATE_OBJ): $(MEX_GATE) $(F_OBJS) | dirs
	$(FC) $(FFLAGS) -DMATLAB_DEFAULT_RELEASE=R2017b -I$(MATLAB_INC) -I$(BUILD_DIR) -c $< -o $@

# ---------- Link with gfortran -shared ----------
$(TARGET): $(MEX_GATE_OBJ) $(ALL_OBJS)
	$(FC) -shared -o $@ $(MEX_GATE_OBJ) $(ALL_OBJS) $(MEXLINKLIBS)

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
	rm -f $(PRIVATE_DIR)/blqmr_.mex*

test:
	matlab -batch "addpath('$(MEX_DIR)'); blqmr_test"