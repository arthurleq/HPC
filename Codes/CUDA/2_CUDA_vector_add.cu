#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>

#include <cuda_runtime.h>

// nvcc -O3 -arch=native 2_CUDA_vector_add.cu -o 2_CUDA_vector_add
// ./2_CUDA_vector_add

// The complete life cycle of a CUDA program, on the simplest example :
// c = a + b for vectors of 32 M floats.
//
//   1. allocate and initialize the data on the host
//   2. allocate memory on the device                  cudaMalloc
//   3. copy the inputs host -> device                 cudaMemcpy(..., cudaMemcpyHostToDevice)
//   4. launch the kernel                              kernel<<<blocks, threads>>>(...)
//   5. copy the result device -> host                 cudaMemcpy(..., cudaMemcpyDeviceToHost)
//   6. free the device memory                         cudaFree
//
// The pointers returned by cudaMalloc are addresses in the memory of the
// GPU : the host must NEVER dereference them (only pass them to kernels
// and to CUDA functions). A common convention : prefix them with d_.

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


// one thread per element. The number of threads is rounded up to a
// multiple of the block size, so the last block may contain threads
// beyond the end of the vectors : they must do nothing.
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// "grid-stride loop" : a grid of ANY size processes vectors of ANY size,
// each thread handling elements i, i + (total number of threads), ...
// The consecutive threads of a warp still access consecutive elements at
// each step. Used when the grid size is chosen from the hardware (e.g. a
// few blocks per multiprocessor) rather than from the data.
__global__ void vector_add_grid_stride(const float* a, const float* b, float* c, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}


int main() {

    const int n = 1 << 25;                  // 32 M elements
    const size_t bytes = n * sizeof(float);  // 128 MB per vector

    // ---- 1. host data ----
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; ++i) {
        a[i] = 1.0f * i;
        b[i] = 2.0f * i;
    }

    // ---- 2. device memory ----
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // GPU timers : events are recorded in the stream of GPU operations,
    // and cudaEventElapsedTime measures the time between two of them
    cudaEvent_t start, after_h2d, after_kernel, after_d2h;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&after_h2d));
    CUDA_CHECK(cudaEventCreate(&after_kernel));
    CUDA_CHECK(cudaEventCreate(&after_d2h));

    CUDA_CHECK(cudaEventRecord(start));

    // ---- 3. inputs host -> device ----
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(after_h2d));

    // ---- 4. the kernel : enough blocks of 256 threads to cover n ----
    // (the block size should be a multiple of the warp size, 32 ;
    // 128 to 512 threads is the usual sweet spot)
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;  // rounded up
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(after_kernel));

    // ---- 5. result device -> host ----
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(after_d2h));
    CUDA_CHECK(cudaEventSynchronize(after_d2h));  // wait until everything is done

    float ms_h2d, ms_kernel, ms_d2h;
    CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, start, after_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, after_h2d, after_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, after_kernel, after_d2h));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (c[i] != 3.0f * i) { ok = false; break; }
    }

    std::cout << "c = a + b with " << n << " floats (" << bytes / 1e6 << " MB per vector) : "
              << (ok ? "correct" : "WRONG") << std::endl
              << std::fixed << std::setprecision(2)
              << "  copy a, b to the GPU  : " << std::setw(7) << ms_h2d << " ms  ("
              << 2 * bytes / (ms_h2d * 1e-3) / 1e9 << " GB/s through PCIe)" << std::endl
              << "  kernel                : " << std::setw(7) << ms_kernel << " ms  ("
              << 3 * bytes / (ms_kernel * 1e-3) / 1e9 << " GB/s in GPU memory)" << std::endl
              << "  copy c back           : " << std::setw(7) << ms_d2h << " ms  ("
              << bytes / (ms_d2h * 1e-3) / 1e9 << " GB/s through PCIe)" << std::endl;

    // ---- the grid-stride version, with a grid sized from the hardware ----
    int device, sm_count;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    CUDA_CHECK(cudaMemset(d_c, 0, bytes));
    CUDA_CHECK(cudaEventRecord(start));
    vector_add_grid_stride<<<4 * sm_count, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(after_kernel));
    CUDA_CHECK(cudaEventSynchronize(after_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&ms_kernel, start, after_kernel));

    CUDA_CHECK(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    ok = true;
    for (int i = 0; i < n; ++i) {
        if (c[i] != 3.0f * i) { ok = false; break; }
    }
    std::cout << "grid-stride kernel, " << 4 * sm_count << " blocks of " << threads
              << " threads : " << (ok ? "correct" : "WRONG") << ", " << ms_kernel << " ms"
              << std::endl;

    // ---- 6. free everything ----
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(after_h2d));
    CUDA_CHECK(cudaEventDestroy(after_kernel));
    CUDA_CHECK(cudaEventDestroy(after_d2h));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}
