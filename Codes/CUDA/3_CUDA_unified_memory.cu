#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>

// nvcc -O3 -arch=native 3_CUDA_unified_memory.cu -o 3_CUDA_unified_memory 
// ./3_CUDA_unified_memory

// UNIFIED MEMORY (or "managed" memory) : cudaMallocManaged returns ONE
// pointer, valid both on the host and on the device. No more cudaMemcpy :
// the driver moves the memory pages on demand. When the GPU touches a page
// that is in the host memory, a PAGE FAULT occurs and the page migrates to
// the GPU (and the other way around).
//
// Very convenient to port a code quickly, or for complex data structures
// (linked lists, trees...) that would be painful to copy by hand. The
// price : page faults are expensive. The solution : tell the driver in
// advance where the data will be needed, with cudaMemPrefetchAsync.
//
// One rule remains : kernel launches are asynchronous, so the host must
// call cudaDeviceSynchronize() before reading data written by a kernel.

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


// migrates [ptr, ptr + bytes) to 'device' (a GPU number, or cudaCpuDeviceId
// for the host) before it is needed. The signature of cudaMemPrefetchAsync
// changed in CUDA 13 : the destination became a cudaMemLocation.
cudaError_t prefetch(const void* ptr, size_t bytes, int device, cudaStream_t stream = 0) {
#if CUDART_VERSION >= 13000
    cudaMemLocation location = {};
    if (device == cudaCpuDeviceId) {
        location.type = cudaMemLocationTypeHost;
    } else {
        location.type = cudaMemLocationTypeDevice;
        location.id = device;
    }
    return cudaMemPrefetchAsync(ptr, bytes, location, 0, stream);
#else
    return cudaMemPrefetchAsync(ptr, bytes, device, stream);
#endif
}


__global__ void scale_add(const float* x, float* y, float a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = a * x[i] + y[i];
    }
}

// runs the kernel and returns its duration in milliseconds
float timed_kernel(const float* x, float* y, int n, int blocks) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    scale_add<<<blocks, 256>>>(x, y, 2.0f, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}


int main() {

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int managed = 0, concurrent = 0, sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    if (!managed) {
        std::cerr << "this GPU does not support managed memory" << std::endl;
        return EXIT_FAILURE;
    }
    // concurrent managed access (Pascal and newer, on Linux) : pages migrate
    // on demand, one by one, through page faults ; prefetching is possible.
    // Without it, all the managed memory is moved at each kernel launch.
    std::cout << "on-demand page migration (concurrent managed access) : "
              << (concurrent ? "yes" : "no") << std::endl;

    const int n = 1 << 25;  // 32 M floats = 128 MB per array
    const size_t bytes = n * sizeof(float);
    const int blocks = 4 * sm_count;

    // ONE pointer per array, usable by the host AND by the device
    float *x = nullptr, *y = nullptr;
    CUDA_CHECK(cudaMallocManaged(&x, bytes));
    CUDA_CHECK(cudaMallocManaged(&y, bytes));

    // ---------------------------------------------------------------
    // PART 1 : the pages follow the accesses
    // ---------------------------------------------------------------
    // initialized by the host : the pages are in the host memory
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0f;
        y[i] = 1.0f;
    }

    // 1st kernel : every page touched by the GPU triggers a page fault and
    // migrates through PCIe, DURING the kernel
    float ms_faults = timed_kernel(x, y, n, blocks);
    // 2nd kernel : the pages are already in the GPU memory
    float ms_resident = timed_kernel(x, y, n, blocks);

    // the host reads the result : the pages of y migrate back (page faults
    // on the host side). Synchronization was done in timed_kernel.
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (y[i] != 5.0f) { ok = false; break; }  // 1 + 2 + 2
    }

    std::cout << std::endl << "=== Part 1 : migration on demand ===" << std::endl
              << std::fixed << std::setprecision(2)
              << "1st kernel (pages migrate during the kernel) : " << std::setw(8)
              << ms_faults << " ms" << std::endl
              << "2nd kernel (pages already on the GPU)        : " << std::setw(8)
              << ms_resident << " ms" << std::endl
              << "result : " << (ok ? "correct" : "WRONG") << std::endl;

    // ---------------------------------------------------------------
    // PART 2 : prefetching
    // ---------------------------------------------------------------
    // the host touches the data again : it comes back to the host memory
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0f;
        y[i] = 1.0f;
    }

    if (concurrent) {
        // move everything to the GPU in big chunks, BEFORE the kernel : much
        // faster than thousands of page faults
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(prefetch(x, bytes, device));
        CUDA_CHECK(prefetch(y, bytes, device));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms_prefetch;
        CUDA_CHECK(cudaEventElapsedTime(&ms_prefetch, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        float ms_kernel = timed_kernel(x, y, n, blocks);

        // and bring the result back to the host before reading it
        CUDA_CHECK(prefetch(y, bytes, cudaCpuDeviceId));
        CUDA_CHECK(cudaDeviceSynchronize());

        ok = true;
        for (int i = 0; i < n; ++i) {
            if (y[i] != 3.0f) { ok = false; break; }  // 1 + 2
        }

        std::cout << std::endl << "=== Part 2 : with prefetching ===" << std::endl
                  << "prefetch x and y to the GPU      : " << std::setw(8) << ms_prefetch
                  << " ms  (" << 2 * bytes / (ms_prefetch * 1e-3) / 1e9 << " GB/s)" << std::endl
                  << "kernel (no page fault anymore)   : " << std::setw(8) << ms_kernel
                  << " ms" << std::endl
                  << "result : " << (ok ? "correct" : "WRONG") << std::endl;
    }

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    return 0;
}
