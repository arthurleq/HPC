# Build all the OpenACC examples of this folder :  make
# Build only one of them :                          make 3_OpenACC_loop_parallelism
# Remove the executables :                          make clean
#
# nvc++ (NVIDIA HPC SDK), the reference OpenACC compiler, is used when it
# is installed, g++ otherwise. To choose explicitly :
#   make COMPILER=nvhpc                    nvc++ -acc=gpu       : NVIDIA GPU
#   make COMPILER=nvhpc TARGET=multicore   nvc++ -acc=multicore : all the CPU cores
#   make COMPILER=gcc                      g++ -fopenacc        : GPU only if gcc was
#                                          built with offloading, a single CPU thread otherwise
 
ifeq ($(shell command -v nvc++ 2> /dev/null),)
  COMPILER ?= gcc
else
  COMPILER ?= nvhpc
endif

TARGET ?= gpu

ifeq ($(COMPILER),nvhpc)
  CXX      := nvc++
  CXXFLAGS ?= -O2 -std=c++17
  # -Minfo=accel : the compiler explains how it parallelized each loop
  ACCFLAGS ?= -acc=$(TARGET) -Minfo=accel
else
  CXX      := g++
  # no -Wextra : gcc reports the parameters only used in OpenACC data
  # clauses as "unused"
  CXXFLAGS ?= -O2 -std=c++17 -Wall
  ACCFLAGS ?= -fopenacc
endif

SOURCES  := $(wildcard *.cpp)
PROGRAMS := $(SOURCES:.cpp=)

all: $(PROGRAMS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $(ACCFLAGS) $< -o $@

clean:
	rm -f $(PROGRAMS)

.PHONY: all clean
