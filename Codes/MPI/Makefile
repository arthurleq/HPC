# Build all the MPI examples of this folder :  make
# Build only one of them :                      make 2_MPI_point_to_point
# Remove the executables :                      make clean
#
# Then run them with mpirun, e.g. :             mpirun -np 4 ./2_MPI_point_to_point

MPICXX   ?= mpicxx
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
# skip the old C++ bindings of MPI (deprecated, removed from the standard
# since MPI-3) : these examples use the C API, which is what C++ codes use
CPPFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX

SOURCES  := $(wildcard *.cpp)
PROGRAMS := $(SOURCES:.cpp=)

all: $(PROGRAMS)

# the hybrid example also uses OpenMP
9_MPI_hybrid_OpenMP: CXXFLAGS += -fopenmp

%: %.cpp
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(PROGRAMS)

.PHONY: all clean
