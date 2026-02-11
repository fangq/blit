# Makefile for BLIT Python extension
#
# Run from python/ directory:
#   make            # Build extension
#   make install    # Install package
#   make test       # Run tests

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
F2PY ?= $(PYTHON) -m numpy.f2py

# Compilers
FC ?= gfortran
CC ?= gcc
FFLAGS ?= -O3 -fPIC -cpp -fopenmp
CFLAGS ?= -O3 -fPIC

# Directories
SRC_DIR = ../src

# Library paths (adjust for your system)
UMFPACK_INC ?= /usr/include/suitesparse
UMFPACK_LIB ?= /usr/lib
LIBS = -lumfpack -lamd -lcholmod -lcolamd -lsuitesparseconfig -llapack -lblas -lgomp

# Sources
FORTRAN_SRC = $(SRC_DIR)/blit_const.f90 \
              $(SRC_DIR)/blit_matrixutil.f90 \
              $(SRC_DIR)/blit_sparseutil.f90 \
              $(SRC_DIR)/blit_ilupcond.f90 \
              $(SRC_DIR)/blit_blqmr.f90 \
              $(SRC_DIR)/blit_blqmr_f2py.f90

C_SRC = $(SRC_DIR)/umf4_f77wrapper.c
C_SRC += $(SRC_DIR)/blit_blas_threads.c

# Signature file for f2py
SIGNATURE = blit_blqmr.pyf

MODULE = _blqmr

# ============================================================================
.PHONY: all build install install-dev wheel test clean help

all: build

# Build using setup.py (recommended)
build: setup.py
	$(PYTHON) setup.py build_ext --inplace
	@echo "Build complete! Run: make test"

# Direct f2py build (faster, for development)
build-direct: $(SIGNATURE)
	$(F2PY) -c $(SIGNATURE) \
		--f90flags="$(FFLAGS)" \
		--f90exec="$(FC)" -I$(UMFPACK_INC) -L$(UMFPACK_LIB) $(LIBS) \
		$(FORTRAN_SRC) $(C_SRC)
	mv $(MODULE)*.so blocksolver/
	@echo "Build complete!"

# Install
install:
	$(PIP) install .

# Editable install
install-dev:
	$(PIP) install -e .

# Build wheel
wheel:
	$(PIP) install build
	$(PYTHON) -m build --wheel

# Test
test: build
	$(PYTHON) -c "import blocksolver; blocksolver.test()"

# Pytest
unittest: build
	$(PYTHON) -m unittest discover -v tests

# Clean
clean:
	rm -rf build/ dist/ *.egg-info/ .eggs/
	rm -f blocksolver/*.so blocksolver/*.pyd
	rm -rf blocksolver/__pycache__ __pycache__ tests/__pycache__
	rm -rf .pytest_cache/

# Help
help:
	@echo "BLIT Python Extension"
	@echo ""
	@echo "Targets:"
	@echo "  make              Build extension (default)"
	@echo "  make build-direct Build with f2py directly"
	@echo "  make install      Install package"
	@echo "  make install-dev  Editable install"
	@echo "  make wheel        Build wheel"
	@echo "  make test         Run tests"
	@echo "  make clean        Clean build files"
	@echo ""
	@echo "Environment:"
	@echo "  UMFPACK_INC=$(UMFPACK_INC)"
	@echo "  UMFPACK_LIB=$(UMFPACK_LIB)"
