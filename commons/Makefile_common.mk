########################################################
#  Makefile for for Blit (derived from the MMC project)
#  Copyright (C) 2009 Qianqian Fang 
#                    <fangq at nmr.mgh.harvard.edu>
#
########################################################

########################################################
# Base Makefile for all example/tests and main program
#
# This file specifies the compiler options for compiling
# and linking
########################################################

ifndef ROOTDIR
	ROOTDIR  := ..
endif

ifndef BLITDIR
	BLITDIR   := $(ROOTDIR)
endif

BLITSRC :=$(BLITDIR)/src

CXX        := g++
AR         := $(CC)
BIN        := $(BLITDIR)/lib
BUILT      := built
BINDIR     := $(BIN)
OBJDIR 	   := $(BUILT)
CCFLAGS    += -c -Wall -g -fno-strict-aliasing #-mfpmath=sse -ffast-math -mtune=core2
INCLUDEDIR := $(BLITDIR)/src -I/usr/include/suitesparse/
EXTRALIB   += -lm
AROUTPUT   += -o
ARFLAGS    :=
MAKE       := make

F90        ?= gfortran
F90OUTDIR      =-J $(OBJDIR)

ifeq ($(F90),g95)
    F90OUTDIR=-fmod=$(OBJDIR)
endif

F90OPT      =-g -cpp -Wextra -Wall -pedantic -O3 -fimplicit-none $(F90OUTDIR) -fbounds-check #-ftrace=full

OPENMP     := -fopenmp
FASTMATH   := #-ffast-math

ECHO	   := echo
MKDIR      := mkdir

DOXY       := doxygen
DOCDIR     := $(BLITDIR)/doc

ifeq ($(CC),icc)
	OPENMP   := -openmp
	FASTMATH :=
	EXTRALIB :=
endif

OBJSUFFIX  := .o
BINSUFFIX  := 

OBJS       := $(addprefix $(OBJDIR)/, $(FILES))
OBJS       := $(subst $(OBJDIR)/$(BLITSRC)/,$(BLITSRC)/,$(OBJS))
OBJS       := $(addsuffix $(OBJSUFFIX), $(OBJS))

BLITLIB_SUFFIX=.so
BINARY=$(BLITLIB_NAME)$(BLITLIB_SUFFIX)
ARFLAGS := -shared -Wl,-soname,$(BINARY).1
F90OPT     +=-fPIC
CCFLAGS    +=-fPIC

ifeq ($(MAKECMDGOALS),static)
    BLITLIB_SUFFIX=.a
    BINARY=$(BLITLIB_NAME)$(BLITLIB_SUFFIX)
    AR         := ar
    ARFLAGS    := r
    AROUTPUT   :=
    F90OPT     +=-fPIC
    CCFLAGS    +=-fPIC
    EXTRALIB    =
endif

TARGETSUFFIX:=$(suffix $(BINARY))

release:   CCFLAGS+= -O3
prof:      CCFLAGS+= -O3 -pg
prof:      ARFLAGS+= -O3 -g -pg

std:    F90OPT+=-std=f95 -fall-intrinsics

all: lib

release prof icc std static lib: $(SUBDIRS) $(BINDIR)/$(BINARY)

$(SUBDIRS):
	$(MAKE) -C $@ --no-print-directory

makedirs:
	@if test ! -d $(OBJDIR); then $(MKDIR) $(OBJDIR); fi
	@if test ! -d $(BINDIR); then $(MKDIR) $(BINDIR); fi

makedocdir:
	@if test ! -d $(DOCDIR); then $(MKDIR) $(DOCDIR); fi

.SUFFIXES : $(OBJSUFFIX) .cpp

##  Compile .cpp files ##
$(OBJDIR)/%$(OBJSUFFIX): %.cpp
	@$(ECHO) Building $@
	$(CXX) $(CCFLAGS) $(USERCCFLAGS) -I$(INCLUDEDIR) -o $@  $<

##  Compile .cpp files ##
%$(OBJSUFFIX): %.cpp
	@$(ECHO) Building $@
	$(CXX) $(CCFLAGS) $(USERCCFLAGS) -I$(INCLUDEDIR) -o $@  $<

##  Compile .f90 files ##
$(OBJDIR)/%$(OBJSUFFIX): %.f90
	$(F90) $(INCLUDEDIRS) $(F90OPT) -c -o $@  $<

##  Compile .c files  ##
$(OBJDIR)/%$(OBJSUFFIX): %.c
	@$(ECHO) Building $@
	$(CC) $(CCFLAGS) $(USERCCFLAGS) -I$(INCLUDEDIR) -o $@  $<

##  Link  ##
$(BINDIR)/$(BINARY): makedirs $(OBJS)
	@$(ECHO) Building $@
	$(AR)  $(ARFLAGS) $(AROUTPUT) $@ $(OBJS) $(EXTRALIB) $(USERARFLAGS)

##  Documentation  ##
doc: makedocdir
	$(DOXY) $(DOXYCFG)

##  Compiler check  ##
g95bin:
	@if [ -z `which ${F90}` ]; then \
		echo "Please first install $F90 and add the path to your PATH environment variable."; exit 1;\
	fi

## Clean
clean:
	-rm -rf $(OBJS) $(OBJDIR) $(BINDIR)
ifdef SUBDIRS
	for i in $(SUBDIRS); do $(MAKE) --no-print-directory -C $$i clean; done
endif

.PHONY: all clean arch makedirs dep $(SUBDIRS)

.DEFAULT_GOAL := lib
