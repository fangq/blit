ROOTDIR = ..

BLITLIB_NAME   ?=libblit

ifeq ($(BUILD_STATIC),1)
    BLITLIB_SUFFIX=.a
endif

FILES?= blit_const blit_matrixutil blit_sparseutil blit_ilupcond umf4_f77wrapper blit_blqmr

include $(ROOTDIR)/commons/Makefile_common.mk

