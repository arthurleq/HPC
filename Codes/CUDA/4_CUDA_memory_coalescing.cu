#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>

#include <cuda_runtime.h>

// nvcc -O3 -arch=native 4_CUDA_memory_coalescing.cu -o 4_CUDA_memory_coalescing 
// ./4_CUDA_memory_coalescing

// Most GPU kernels are limited by the memory bandwidth, not by the
// computation. HOW the threads access the memory then matters as much as
// how much they access.
//
// COALESCING : the 32 threads of a warp execute the same load instruction
// at the same time. If they read 32 CONSECUTIVE floats (128 bytes), the
// hardware merges the 32 accesses into a few wide memory transactions. If
// their addresses are far apart (a "strided" access), each thread needs
// its own transaction, which brings 32 bytes to use only 4 of them :
//
//   coalesced : thread  0  1  2  3 ... 31           strided : thread 0    1    2
//               address 0  4  8 12 ... 124                  address 0  4096 8192 ...
//               -> 4 transactions of 32 bytes                -> 32 transactions
//
// Test case : the matrix TRANSPOSE, out[j][i] = in[i][j]. Reading a row of
// 'in' is contiguous, but it becomes a COLUMN of 'out', which is strided.
// One of the two accesses is necessarily bad... unless the tile goes
// through the SHARED MEMORY : a small (48-228 KB per SM) memory, on the
// chip, ~100 times faster than the global memory, shared by the threads of
// a block and managed by hand. It is used here to reorder the accesses.

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

// Each block handles a TILE x TILE tile of the matrix with TILE x ROWS
// threads : each thread processes TILE / ROWS = 4 elements of the tile.
const int TILE = 32;
const int ROWS = 8;


// REFERENCE : a plain copy, coalesced for the reads and the writes. No
// transpose can be faster : it gives the bandwidth to aim for.
__global__ void copy(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;  // column
    int y = blockIdx.y * TILE + threadIdx.y;  // row
    for (int k = 0; k < TILE; k += ROWS) {
        out[(y + k) * n + x] = in[(y + k) * n + x];
    }
}

// NAIVE transpose : the reads are coalesced (consecutive threads, consecutive
// x, same row), the writes are strided (consecutive x = consecutive ROWS
// of out, n floats apart).
__global__ void transpose_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int k = 0; k < TILE; k += ROWS) {
        out[x * n + (y + k)] = in[(y + k) * n + x];
    }
}

// SHARED MEMORY transpose : the tile is read row by row (coalesced) into
// shared memory, then written row by row (coalesced) into the transposed
// position of the output. The transposition itself happens inside the
// shared memory, where strided accesses are cheap... almost (see below).
__global__ void transpose_shared(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int k = 0; k < TILE; k += ROWS) {
        tile[threadIdx.y + k][threadIdx.x] = in[(y + k) * n + x];
    }

    // wait until the WHOLE tile is in shared memory : the elements a
    // thread writes below were loaded by OTHER threads of the block
    __syncthreads();

    // the transposed tile goes to block (blockIdx.y, blockIdx.x)
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int k = 0; k < TILE; k += ROWS) {
        out[(y + k) * n + x] = tile[threadIdx.x][threadIdx.y + k];
    }
}

// The shared memory is divided into 32 BANKS (consecutive 4-byte words go
// to consecutive banks). The threads of a warp can access 32 different
// banks at the same time, but accesses to the SAME bank are serialized
// (a "bank conflict"). Reading a COLUMN of tile[32][32] means reading
// words 32 apart : all in the same bank -> a 32-way conflict.
// One extra column, tile[32][33], shifts each row by one bank : the
// elements of a column now fall in 32 different banks.
__global__ void transpose_shared_padded(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int k = 0; k < TILE; k += ROWS) {
        tile[threadIdx.y + k][threadIdx.x] = in[(y + k) * n + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int k = 0; k < TILE; k += ROWS) {
        out[(y + k) * n + x] = tile[threadIdx.x][threadIdx.y + k];
    }
}


int main() {

    const int n = 4096;  // a 4096 x 4096 matrix of floats (64 MB), a multiple of TILE
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    std::vector<float> in(static_cast<size_t>(n) * n), out(static_cast<size_t>(n) * n);
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(i);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid(n / TILE, n / TILE);
    dim3 block(TILE, ROWS);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "transpose of a " << n << " x " << n << " matrix (effective bandwidth = "
              << "bytes read + written / time)" << std::endl;

    auto run = [&](const char* name, void (*kernel)(const float*, float*, int), bool transposed) {
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        kernel<<<grid, block>>>(d_in, d_out, n);  // warm-up
        CUDA_CHECK(cudaGetLastError());

        const int repetitions = 20;
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < repetitions; ++r) kernel<<<grid, block>>>(d_in, d_out, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= repetitions;

        CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
        bool ok = true;
        for (int i = 0; i < n && ok; ++i) {
            for (int j = 0; j < n; ++j) {
                float expected = transposed ? in[static_cast<size_t>(j) * n + i]
                                            : in[static_cast<size_t>(i) * n + j];
                if (out[static_cast<size_t>(i) * n + j] != expected) { ok = false; break; }
            }
        }
        std::cout << name << std::fixed << std::setprecision(3) << std::setw(7) << ms
                  << " ms | " << std::setw(7) << std::setprecision(1)
                  << 2 * bytes / (ms * 1e-3) / 1e9 << " GB/s | " << (ok ? "correct" : "WRONG")
                  << std::endl;
    };

    run("copy (reference)                 : ", copy, false);
    run("naive transpose                  : ", transpose_naive, true);
    run("shared memory                    : ", transpose_shared, true);
    run("shared memory, no bank conflicts : ", transpose_shared_padded, true);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
