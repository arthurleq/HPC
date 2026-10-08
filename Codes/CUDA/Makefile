# Build all the CUDA examples of this folder :  make
# Build only one of them :                       make 6_CUDA_reduction
# Remove the executables :                       make clean
#
# By default, the code is compiled for the GPU of this machine
# (-arch=native). On a machine without GPU (e.g. the login node of a
# cluster), give the compute capability of the target GPU :
#   make CUDA_ARCH=sm_80     # A100
#   make CUDA_ARCH=sm_90     # H100
#   make CUDA_ARCH=sm_86     # RTX 30xx, A40...
#   make CUDA_ARCH=sm_89     # RTX 40xx, L40...

NVCC      ?= nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH)

SOURCES  := $(wildcard *.cu)
PROGRAMS := $(SOURCES:.cu=)

all: $(PROGRAMS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(PROGRAMS)

.PHONY: all clean
