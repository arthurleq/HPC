#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>  // CUB : the library of parallel primitives shipped with CUDA

// nvcc -O3 -arch=native 6_CUDA_reduction.cu -o 6_CUDA_reduction
// ./6_CUDA_reduction

// REDUCTION : sum the 64 M integers of an array. Trivial sequentially, it
// is THE classic exercise of GPU programming, because millions of threads
// must combine their values into one. Four versions, from the worst to
// the best :
//
//   1. every thread does atomicAdd on ONE global counter
//      -> 64 M atomic operations on the same address, serialized
//
//   2. tree reduction in SHARED MEMORY inside each block, then ONE atomicAdd
//      per block :
//         256 values -> 128 partial sums -> 64 -> ... -> 1   (8 steps)
//
//   3. WARP SHUFFLES : the 32 threads of a warp exchange values directly
//      between their registers (__shfl_down_sync), without shared memory
//      nor __syncthreads. Plus each thread first sums MANY elements on its
//      own (grid-stride loop) : fewer blocks, fewer atomics, more work per
//      thread.
//
//   4. a library : cub::DeviceReduce::Sum. In real codes, use the
//      libraries (CUB, Thrust, cuBLAS...) : tuned for every GPU generation.

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d : %s\n",              \
                         cudaGetErrorName(err_), __FILE__, __LINE__,           \
                         cudaGetErrorString(err_));                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

const int BLOCK = 256;  // threads per block (a power of 2 for version 2)


// VERSION 1 : one atomic per element
__global__ void reduce_atomic(const int* in, int* sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(sum, in[i]);
    }
}

// VERSION 2 : tree reduction in shared memory
//
//   step 1 : s = 128   cache[t] += cache[t + 128]   for t < 128
//   step 2 : s = 64    cache[t] += cache[t + 64]    for t < 64
//   ...
//   step 8 : s = 1     cache[0] += cache[1]         -> the sum of the block
//
// The threads that work at each step are CONTIGUOUS (t < s) : whole warps
// work or rest together, and the accesses cache[t], cache[t + s] hit
// different shared memory banks (no bank conflict).
__global__ void reduce_shared(const int* in, int* sum, int n) {
    __shared__ int cache[BLOCK];

    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    cache[t] = (i < n) ? in[i] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (t < s) {
            cache[t] += cache[t + s];
        }
        __syncthreads();  // each step needs the results of the previous one
    }

    if (t == 0) {
        atomicAdd(sum, cache[0]);  // one atomic per block
    }
}

// VERSION 3 : warp shuffles.
// __shfl_down_sync(mask, v, offset) gives each thread the value v of the
// thread 'offset' lanes further in the same warp (mask = the threads
// taking part : all 32 here). After 5 steps, lane 0 holds the warp's sum :
//
//   offset 16 : lane i += lane i+16      offset 8 : lane i += lane i+8
//   ... offset 1 : lane 0 = sum of the 32 lanes
__device__ int warp_sum(int v) {
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__global__ void reduce_warp(const int* in, int* sum, int n) {

    // 1. each thread sums many elements on its own, in a register
    int v = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        v += in[i];
    }

    // 2. sum inside each warp
    v = warp_sum(v);

    // 3. the first warp sums the partial sums of the warps of the block
    __shared__ int warp_sums[32];  // at most 1024 threads = 32 warps per block
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();

    if (warp == 0) {
        int nwarps = blockDim.x / 32;
        v = (lane < nwarps) ? warp_sums[lane] : 0;
        v = warp_sum(v);
        if (lane == 0) {
            atomicAdd(sum, v);  // one atomic per block, and few blocks
        }
    }
}


int main() {

    const int n = 1 << 26;  // 64 M integers = 256 MB
    const size_t bytes = n * sizeof(int);

    // values from 0 to 9 : the sum (~ 300 M) fits in an int, and integer
    // sums are exact whatever the order of the additions
    std::vector<int> h(n);
    long long expected = 0;
    for (int i = 0; i < n; ++i) {
        h[i] = (i * 7) % 10;
        expected += h[i];
    }

    int *d_in = nullptr, *d_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));

    int device, sm_count;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    // temporary storage for CUB : a first call with a null pointer only
    // computes the size needed
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_sum, n));
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "sum of " << n << " integers (expected " << expected << ")" << std::endl;

    // 'launch' runs one version ; the counter is reset before each run
    auto run = [&](const char* name, auto launch) {
        CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(int)));
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        int result;
        CUDA_CHECK(cudaMemcpy(&result, d_sum, sizeof(int), cudaMemcpyDeviceToHost));
        std::cout << name << std::fixed << std::setprecision(3) << std::setw(8) << ms << " ms | "
                  << std::setw(7) << std::setprecision(1) << bytes / (ms * 1e-3) / 1e9
                  << " GB/s | sum = " << result << (result == expected ? "  correct" : "  WRONG")
                  << std::endl;
    };

    const int blocks = (n + BLOCK - 1) / BLOCK;

    run("1. atomicAdd per element     : ", [&] {
        reduce_atomic<<<blocks, BLOCK>>>(d_in, d_sum, n);
    });
    run("2. shared memory tree        : ", [&] {
        reduce_shared<<<blocks, BLOCK>>>(d_in, d_sum, n);
    });
    run("3. warp shuffles, grid-stride : ", [&] {
        reduce_warp<<<8 * sm_count, BLOCK>>>(d_in, d_sum, n);
    });
    run("4. cub::DeviceReduce::Sum    : ", [&] {
        CUDA_CHECK(cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_sum, n));
    });

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_sum));
    return 0;
}
