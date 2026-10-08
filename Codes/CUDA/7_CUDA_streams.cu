#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

#include <cuda_runtime.h>

// nvcc -O3 -arch=native 7_CUDA_streams.cu -o 7_CUDA_streams 
// ./7_CUDA_streams
// nsys profile ./7_CUDA_streams    shows the overlap on a timeline (Nsight Systems)

// Overlapping data transfers and computations : the same idea as
// ../OpenACC/5_OpenACC_async.cpp, written by hand. Two ingredients :
//
// 1. PINNED host memory. Memory from new / malloc is "pageable" : the OS
//    may move its pages at any time. The copy engine of the GPU (DMA) can
//    only read page-locked ("pinned") memory, so the driver first copies
//    pageable data into an internal pinned buffer, then transfers it.
//    cudaMallocHost allocates pinned memory directly : faster transfers,
//    and REQUIRED for truly asynchronous copies. It can't be swapped out,
//    so don't pin gigabytes of it.
//
// 2. STREAMS. A stream is a queue of GPU operations (copies, kernels)
//    executed in order ; operations of DIFFERENT streams may run at the
//    same time. A GPU has separate engines for the copies host -> device,
//    device -> host and for the kernels : cutting the data into chunks,
//    each one handled by a stream, keeps all of them busy :
//
//   1 stream  : [ H->D everything ][ kernel on everything ][ D->H everything ]
//
//   4 streams : [H->D 0][kernel 0][D->H 0]
//                       [H->D 1][kernel 1][D->H 1]
//                               [H->D 2][kernel 2][D->H 2]
//                                       [H->D 3][kernel 3][D->H 3]

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

double now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// some work on each value : 100 iterations of the logistic map, enough
// computation to be comparable to the time of the transfers
__global__ void work(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float r = x[i];
        float v = 0.5f;
        for (int k = 0; k < 100; ++k) v = r * v * (1.0f - v);
        y[i] = v;
    }
}


int main() {

    const int n = 1 << 25;  // 32 M floats = 128 MB
    const size_t bytes = n * sizeof(float);
    const int threads = 256;

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    // ---------------------------------------------------------------
    // PART 1 : pageable vs pinned memory
    // ---------------------------------------------------------------
    std::vector<float> pageable(n, 1.0f);
    float *pinned_x = nullptr, *pinned_y = nullptr;
    CUDA_CHECK(cudaMallocHost(&pinned_x, bytes));  // pinned host memory
    CUDA_CHECK(cudaMallocHost(&pinned_y, bytes));
    for (int i = 0; i < n; ++i) pinned_x[i] = 2.5f + 1.5f * i / n;

    CUDA_CHECK(cudaMemcpy(d_x, pageable.data(), bytes, cudaMemcpyHostToDevice));  // warm-up

    double t0 = now();
    CUDA_CHECK(cudaMemcpy(d_x, pageable.data(), bytes, cudaMemcpyHostToDevice));
    double t_pageable = now() - t0;

    t0 = now();
    CUDA_CHECK(cudaMemcpy(d_x, pinned_x, bytes, cudaMemcpyHostToDevice));
    double t_pinned = now() - t0;

    std::cout << "=== Part 1 : copy of " << bytes / 1e6 << " MB to the GPU ===" << std::endl
              << std::fixed << std::setprecision(2)
              << "from pageable memory : " << std::setw(7) << t_pageable * 1e3 << " ms  ("
              << bytes / t_pageable / 1e9 << " GB/s)" << std::endl
              << "from pinned memory   : " << std::setw(7) << t_pinned * 1e3 << " ms  ("
              << bytes / t_pinned / 1e9 << " GB/s)" << std::endl;

    // ---------------------------------------------------------------
    // PART 2 : one stream vs several streams
    // ---------------------------------------------------------------
    std::cout << std::endl << "=== Part 2 : copy in + kernel + copy out ===" << std::endl;

    // warm-up : the code of a kernel is loaded on the GPU at its first
    // launch ("lazy loading"), which would distort the first measurement
    work<<<1, threads>>>(d_x, d_y, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // reference : everything in sequence (the default stream)
    t0 = now();
    CUDA_CHECK(cudaMemcpy(d_x, pinned_x, bytes, cudaMemcpyHostToDevice));
    work<<<(n + threads - 1) / threads, threads>>>(d_x, d_y, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(pinned_y, d_y, bytes, cudaMemcpyDeviceToHost));
    double t_sequential = now() - t0;
    std::vector<float> reference(pinned_y, pinned_y + n);
    std::cout << "1 stream             : " << std::setw(7) << t_sequential * 1e3 << " ms"
              << std::endl;

    const int nstreams = 4;
    cudaStream_t streams[nstreams];
    for (int s = 0; s < nstreams; ++s) CUDA_CHECK(cudaStreamCreate(&streams[s]));

    for (int nchunks : {4, 16}) {
        for (int i = 0; i < n; ++i) pinned_y[i] = 0.0f;

        t0 = now();
        for (int c = 0; c < nchunks; ++c) {
            int first = static_cast<int>(static_cast<long long>(n) * c / nchunks);
            int count = static_cast<int>(static_cast<long long>(n) * (c + 1) / nchunks) - first;
            cudaStream_t s = streams[c % nstreams];

            // all three operations are queued in stream s and the host goes
            // on immediately : cudaMemcpyAsync needs PINNED host memory
            CUDA_CHECK(cudaMemcpyAsync(d_x + first, pinned_x + first, count * sizeof(float),
                                       cudaMemcpyHostToDevice, s));
            work<<<(count + threads - 1) / threads, threads, 0, s>>>(d_x + first, d_y + first,
                                                                     count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(pinned_y + first, d_y + first, count * sizeof(float),
                                       cudaMemcpyDeviceToHost, s));
        }
        CUDA_CHECK(cudaDeviceSynchronize());  // wait for all the streams
        double t = now() - t0;

        bool same = true;
        for (int i = 0; i < n; ++i) {
            if (pinned_y[i] != reference[i]) { same = false; break; }
        }
        std::cout << nstreams << " streams, " << std::setw(2) << nchunks << " chunks : "
                  << std::setw(7) << t * 1e3 << " ms | same result : " << (same ? "yes" : "NO")
                  << std::endl;
    }

    for (int s = 0; s < nstreams; ++s) CUDA_CHECK(cudaStreamDestroy(streams[s]));
    CUDA_CHECK(cudaFreeHost(pinned_x));
    CUDA_CHECK(cudaFreeHost(pinned_y));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    return 0;
}
