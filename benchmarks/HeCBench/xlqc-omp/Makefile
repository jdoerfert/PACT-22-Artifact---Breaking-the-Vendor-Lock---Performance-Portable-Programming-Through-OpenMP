#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
GSL_INC   =-I/path/to/gsl-2.7/include
GSL_LIB   =-L/path/to/gsl-2.7/lib -lgsl -lgslcblas 


#===============================================================================
# Program name & source code list
#===============================================================================

program = main

obj=basis.o scf.o main.o crys.o cints.o


#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++14 -Wall -I../xlqc-cuda -I../xlqc-cuda/int_lib $(GSL_INC)

# Linker Flags
LDFLAGS = $(GSL_LIB)

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

scf.o: ../xlqc-cuda/scf.cc
	$(CC) -c -o $@ $< $(CFLAGS)

basis.o: ../xlqc-cuda/basis.cc
	$(CC) -c -o $@ $< $(CFLAGS)

crys.o: ../xlqc-cuda/int_lib/crys.cc
	$(CC) -c -o $@ $< $(CFLAGS)

cints.o: ../xlqc-cuda/int_lib/cints.cc
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) sp
	./$(program) dp

