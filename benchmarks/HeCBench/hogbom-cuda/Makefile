#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
VERIFY    = no

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

obj = main.o kernels.o reference.o timer.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++14 -Xcompiler -Wall -arch=$(ARCH)

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(VERIFY),yes)
  CFLAGS += -DVERIFY
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cpp utils.h timer.h reference.h kernels.h
	$(CC) $(CFLAGS) -c $< -o $@

reference.o: reference.cpp reference.h 
	$(CC) $(CFLAGS) -c $< -o $@

timer.o: timer.cpp timer.h 
	$(CC) $(CFLAGS) -c $< -o $@

kernels.o: kernels.cu kernels.h 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) data/dirty_4096.img data/psf_4096.img 1000
