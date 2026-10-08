# Build all the OpenMP examples of this folder :  make
# Build only one of them :                         make 6_OpenMP_tasks
# Remove the executables :                         make clean
#
# The compiler and the flags can be changed from the command line, e.g.
#   make CXX=clang++
#   make CXXFLAGS="-O3 -march=native"

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
OMPFLAGS ?= -fopenmp

SOURCES  := $(wildcard *.cpp)
PROGRAMS := $(SOURCES:.cpp=)

all: $(PROGRAMS)

# the SIMD example needs the widest vector instructions of the processor
7_OpenMP_simd_collapse: CXXFLAGS += -march=native

%: %.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

clean:
	rm -f $(PROGRAMS)

.PHONY: all clean
