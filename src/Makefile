ROOTDIR = ..

BLITLIB_NAME   ?=libblit

ifeq ($(BUILD_STATIC),1)
    BLITLIB_SUFFIX=.a
endif

FILES?= blit_const blit_matrixutil blit_sparseutil blit_ilupcond umf4_f77wrapper blit_blas_threads blit_blqmr

# OpenMP support - enable by default, disable with USE_OPENMP=0
USE_OPENMP ?= 1

ifeq ($(USE_OPENMP),1)
    OMPFLAG ?= -fopenmp
    FCFLAGS += $(OMPFLAG)
    LDFLAGS += $(OMPFLAG)
endif

include $(ROOTDIR)/commons/Makefile_common.mk
