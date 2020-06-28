FILES ?=passdata passdataplus blqmr_test

BINDIR=bin
MKDIR =mkdir
CC   ?=gcc
CXX  ?=g++
INCLUDEDIR=../../include
LINKDIR=../../lib
LINKLIB=-lblit -lumfpack -llapack -lm -lgfortran
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
