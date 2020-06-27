ROOTDIR = ../

BLITLIB_SUFFIX :=so
BLITLIB_NAME   ?=libblqmr

ifeq ($(BUILD_STATIC),1)
    BLITLIB_SUFFIX=a
endif

BINARY=$(BLITLIB_NAME).$(BLITLIB_SUFFIX)

FILES= blit_const blit_matrixutil blit_sparseutil blit_ilupcond umf4_f77wrapper blit_blqmr

include $(ROOTDIR)/commons/Makefile_common.mk

