# =============================================================================
# Makefile for BLIT BLQMR MEX extension
#
# Usage:
#   make              - build MEX file (auto-detect platform)
#   make clean        - remove build artifacts
#   make test         - run MATLAB test
#   make debug        - build with debug flags
#
# Environment variables:
#   UMFPACK_INC       - SuiteSparse include path(s)
#   UMFPACK_LIB       - SuiteSparse library path(s)
#   OPENBLAS_DIR      - OpenBLAS installation prefix
#
# This Makefile lives in blit/matlab/src/
# =============================================================================

# ---------- Platform detection ----------
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  PLATFORM := macos
else ifeq ($(UNAME),Linux)
  PLATFORM := linux
else
  PLATFORM := windows
endif

# ---------- Directory layout ----------
MKFILE_DIR  := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
MEX_DIR     := $(abspath $(MKFILE_DIR)/..)
ROOT_DIR    := $(abspath $(MEX_DIR)/..)
SRC_DIR     := $(ROOT_DIR)/src
PRIVATE_DIR := $(MEX_DIR)/private
BUILD_DIR   := $(MKFILE_DIR)/build

# ---------- MATLAB ----------
MEX         := mex

ifeq ($(PLATFORM),macos)
  MEX_EXT   := $(shell mexext 2>/dev/null || echo mexmaca64)
else ifeq ($(PLATFORM),linux)
  MEX_EXT   := $(shell mexext 2>/dev/null || echo mexa64)
else
  MEX_EXT   := $(shell mexext 2>/dev/null || echo mexw64)
endif

# ---------- Compilers ----------
FC  ?= gfortran
CC  ?= gcc

# ---------- Flags ----------
FFLAGS_COMMON := -O3 -cpp -fPIC -fopenmp
CFLAGS_COMMON := -O3 -fPIC

ifeq ($(PLATFORM),macos)
  FFLAGS  := $(FFLAGS_COMMON)
  CFLAGS  := $(CFLAGS_COMMON)
else ifeq ($(PLATFORM),linux)
  FFLAGS  := $(FFLAGS_COMMON)
  CFLAGS  := $(CFLAGS_COMMON)
else
  FFLAGS  := -O3 -cpp -fopenmp
  CFLAGS  := -O3
endif

ifdef DEBUG
  FFLAGS := $(subst -O3,-O0 -g -fcheck=all -fbacktrace,$(FFLAGS))
  CFLAGS := $(subst -O3,-O0 -g,$(CFLAGS))
endif

# ---------- SuiteSparse / UMFPACK ----------
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

# ---------- BLAS / LAPACK ----------
# Link system BLAS/LAPACK (same as the standalone C++ build).
# Inside MATLAB, these symbols get resolved to MATLAB's MKL at runtime.
# The MEX gateway calls maxNumCompThreads(1) to prevent MKL's threaded
# dgemm from crashing due to stack overflow.
BLAS_INC :=
BLAS_LIB := -llapack -lblas

# ---------- UMFPACK libraries ----------
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

# ---------- OpenMP ----------
# Use system libgomp (same as standalone build).
ifeq ($(PLATFORM),macos)
  OMP_LIB   := -lomp
else
  OMP_LIB   := -lgomp
endif

# ---------- MEX link libraries ----------
# -Wl,-Bdynamic forces shared .so linking (system .a files lack -fPIC).
# Uses MATLAB's libmwblas/libmwlapack and libiomp5 to avoid runtime conflicts.
MEXLINKLIBS = $(SS_LIB_FLAGS) -Wl,-Bdynamic $(UMFPACK_LIBS) $(BLAS_LIB) -lgfortran $(OMP_LIB) -lm

# ---------- Source files ----------
F_SRCS := $(SRC_DIR)/blit_const.f90 \
          $(SRC_DIR)/blit_matrixutil.f90 \
          $(SRC_DIR)/blit_sparseutil.f90 \
          $(SRC_DIR)/blit_ilupcond.f90 \
          $(SRC_DIR)/blit_blqmr.f90 \
          $(SRC_DIR)/blit_blqmr_f2py.f90

C_SRCS := $(SRC_DIR)/umf4_f77wrapper.c \
          $(SRC_DIR)/blit_blas_threads.c

MEX_GATE := $(MKFILE_DIR)/blqmr_mex.c

# ---------- Object files ----------
F_OBJS := $(patsubst $(SRC_DIR)/%.f90,$(BUILD_DIR)/%.o,$(F_SRCS))
C_OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SRCS))
ALL_OBJS := $(F_OBJS) $(C_OBJS)

# ---------- Target ----------
TARGET := $(PRIVATE_DIR)/blqmr_.$(MEX_EXT)

# =============================================================================
# Rules
# =============================================================================

.PHONY: all clean test debug dirs

all: dirs $(TARGET)

debug:
	$(MAKE) DEBUG=1

dirs:
	@mkdir -p $(BUILD_DIR) $(PRIVATE_DIR)

# ---------- Link MEX ----------
$(TARGET): $(ALL_OBJS) $(MEX_GATE)
	$(MEX) CC='$(CC)' \
	  LINKLIBS='-L"$$MATLABROOT/bin/$$ARCH" -lmx -lmex $(MEXLINKLIBS)' \
	  $(SS_INC_FLAGS) $(BLAS_INC) \
	  $(MEX_GATE) $(ALL_OBJS) \
	  -outdir $(PRIVATE_DIR) -output blqmr_

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

$(BUILD_DIR)/blit_blqmr_f2py.o: $(SRC_DIR)/blit_blqmr_f2py.f90 $(BUILD_DIR)/blit_blqmr.o
	$(FC) $(FFLAGS) -I$(BUILD_DIR) -J$(BUILD_DIR) -c $< -o $@

# ---------- C objects ----------
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | dirs
	$(CC) $(CFLAGS) $(SS_INC_FLAGS) -c $< -o $@

# ---------- Housekeeping ----------
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(PRIVATE_DIR)/blqmr_.mex*

test:
	matlab -batch "addpath('$(MEX_DIR)'); blqmr_test"