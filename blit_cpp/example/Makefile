FILES ?=passdata passdataplus blqmr_test

BINDIR=bin
MKDIR =mkdir
CC   ?=gcc
CXX  ?=g++
INCLUDEDIR=../include
LINKDIR=../../blit_f90/src/blqmr/bin/
LINKLIB=-lblit_gfortran -lumfpack -llapack
CCFLAGS=-g
ECHO=echo

OBJS       := $(addprefix $(BINDIR)/, $(FILES))

all: makedirs $(OBJS)

makedirs:
	@if test ! -d $(BINDIR); then $(MKDIR) $(BINDIR); fi

$(BINDIR)/%: %.c
	@$(ECHO) Building $@
	$(CC) $(CCFLAGS) -L$(LINKDIR) -I$(INCLUDEDIR) -o $@ $< $(LINKLIB)

$(BINDIR)/%: %.cpp
	@$(ECHO) Building $@
	$(CXX) $(CCFLAGS) -L$(LINKDIR) -I$(INCLUDEDIR) -o $@ $< $(LINKLIB)

clean:
	rm -rf $(BINDIR)
