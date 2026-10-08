#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

#include <cuda_runtime.h>

// nvcc -O3 -arch=native 5_CUDA_shared_memory_matmul.cu -o 5_CUDA_shared_memory_matmul 
// ./5_CUDA_shared_memory_matmul

// Matrix product C = A x B (n x n floats, stored row by row) : each C[i][j]
// is the dot product of row i of A and column j of B.
//
// NAIVE kernel : one thread per element of C, reading its row of A and its
// column of B from the global memory : 2n loads for 2n floating-point
// operations. The same elements are read again and again by different
// threads (row i of A is read by the n threads of row i of C) : the kernel
// is limited by the memory traffic.
//
// TILED kernel : a block of TILE x TILE threads computes a TILE x TILE tile
// of C. The dot products are cut into chunks of TILE ; for each chunk, the
// block loads ONE tile of A and ONE tile of B into SHARED MEMORY (each
// thread loads one element of each), then every thread reads them TILE
// times from the fast shared memory :
//
//            B tile t
//              | |
//   A tile t ->[C tile]       C tile = sum over t of (A tile t) x (B tile t)
//
// -> TILE times fewer loads from the global memory. Shared memory is the
// main tool to turn a memory-bound kernel into a compute-bound one.

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

const int TILE = 16;  // blocks of 16 x 16 = 256 threads


__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // consecutive threads -> consecutive columns
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {

    // shared by the 256 threads of the block
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;

    for (int t = 0; t < (n + TILE - 1) / TILE; ++t) {

        // each thread loads one element of the A tile and one of the B tile
        // (0 outside of the matrices, when n is not a multiple of TILE)
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;
        As[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;

        // barrier 1 : the tiles must be COMPLETE before anybody uses them
        // (each thread reads elements loaded by other threads)
        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // barrier 2 : everybody must be done with the tiles before the next
        // iteration overwrites them
        __syncthreads();
    }
    // (__syncthreads must be reached by ALL the threads of the block : never
    // put it inside a condition that only some threads satisfy)

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}


int main() {

    const int n = 2048;
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    const double gflop = 2.0 * n * n * n / 1e9;

    // positive values : no cancellation in the dot products, so that the
    // relative error measures the rounding of the float sums only
    std::vector<float> A(static_cast<size_t>(n) * n), B(A.size()), C(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = static_cast<float>((i * 7) % 13) / 13.0f;
        B[i] = static_cast<float>((i * 11) % 17) / 17.0f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "C = A x B, n = " << n << " (" << gflop << " GFLOP)" << std::endl;

    auto run = [&](const char* name, void (*kernel)(const float*, const float*, float*, int)) {
        CUDA_CHECK(cudaMemset(d_C, 0, bytes));
        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<grid, block>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        // check 1000 elements against a double precision computation on the
        // host (the whole product would take too long on the CPU)
        CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        double max_error = 0.0;
        for (int s = 0; s < 1000; ++s) {
            int i = (s * 7919) % n, j = (s * 104729) % n;
            double exact = 0.0;
            for (int k = 0; k < n; ++k) {
                exact += static_cast<double>(A[static_cast<size_t>(i) * n + k]) *
                         B[static_cast<size_t>(k) * n + j];
            }
            max_error = std::fmax(max_error,
                                  std::fabs(C[static_cast<size_t>(i) * n + j] - exact) / exact);
        }

        std::cout << name << std::fixed << std::setprecision(2) << std::setw(8) << ms
                  << " ms | " << std::setw(8) << std::setprecision(1) << gflop / (ms * 1e-3)
                  << " GFLOP/s | max relative error " << std::scientific << std::setprecision(1)
                  << max_error << (max_error < 1e-3 ? "  correct" : "  WRONG") << std::endl;
    };

    run("naive (global memory only) : ", matmul_naive);
    run("tiled (shared memory)      : ", matmul_tiled);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
