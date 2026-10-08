#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

// nvcc -O3 -arch=native 1_CUDA_hello_world.cu -o 1_CUDA_hello_world
// ./1_CUDA_hello_world
// (-arch=native compiles for the GPU of this machine ; on a cluster, give
// the architecture explicitly, e.g. -arch=sm_80 for an A100, sm_90 for an H100)

// CUDA ("Compute Unified Device Architecture") is NVIDIA's platform to
// program its GPUs. Unlike OpenACC, nothing is automatic : the programmer
// writes the functions that run on the GPU (the KERNELS), decides how many
// threads execute them, and manages the GPU memory by hand.
//
// A kernel is executed by a GRID of BLOCKS of THREADS :
//
//   grid  (all the threads of one kernel launch)
//   +--------------------+--------------------+--------------------+
//   | block 0            | block 1            | block 2            |
//   | t0 t1 t2 ... t255  | t0 t1 t2 ... t255  | t0 t1 t2 ... t255  |
//   +--------------------+--------------------+--------------------+
//
//   - the threads of a block run on the same multiprocessor (SM), can
//     share a fast on-chip memory (shared memory) and synchronize
//   - the blocks are independent : they can run in any order, on any SM
//   - the threads are executed by groups of 32, the WARPS : the 32 threads
//     of a warp execute the same instruction at the same time (SIMT :
//     Single Instruction, Multiple Threads, an extension of SIMD)


// every CUDA function returns an error code : ALWAYS check it. Kernel
// launches return nothing : their errors are retrieved with
// cudaGetLastError() (launch errors) and cudaDeviceSynchronize()
// (errors that happened during the execution).
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


// ---------------------------------------------------------------
// PART 1 : what GPU do we have ?
// ---------------------------------------------------------------
void print_device_properties() {

    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    std::cout << "number of CUDA devices : " << count << std::endl;

    for (int dev = 0; dev < count; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        // the memory clock (kHz) and the bus width (bits) give the
        // theoretical bandwidth : 2 transfers per clock (double data rate)
        int memory_clock_khz = 0, bus_width_bits = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
        double bandwidth = 2.0 * memory_clock_khz * 1e3 * (bus_width_bits / 8) / 1e9;

        std::cout << "device " << dev << " : " << prop.name << std::endl
                  << "  compute capability        : " << prop.major << "." << prop.minor
                  << "   (the GPU generation, -arch=sm_" << prop.major << prop.minor << ")"
                  << std::endl
                  << "  multiprocessors (SM)      : " << prop.multiProcessorCount << std::endl
                  << "  warp size                 : " << prop.warpSize << " threads" << std::endl
                  << "  max threads per block     : " << prop.maxThreadsPerBlock << std::endl
                  << "  max threads per SM        : " << prop.maxThreadsPerMultiProcessor
                  << std::endl
                  << "  shared memory per block   : " << prop.sharedMemPerBlock / 1024 << " KB"
                  << std::endl
                  << "  global memory             : " << prop.totalGlobalMem / (1 << 20) << " MB"
                  << std::endl
                  << "  L2 cache                  : " << prop.l2CacheSize / 1024 << " KB"
                  << std::endl
                  << "  theoretical bandwidth     : " << bandwidth << " GB/s" << std::endl;
    }
}


// ---------------------------------------------------------------
// PART 2 : a first kernel
// ---------------------------------------------------------------
// __global__ : a kernel, called from the host, executed on the device.
// Each thread knows its position through built-in variables :
//   threadIdx.x  its index in its block      (0 .. blockDim.x - 1)
//   blockIdx.x   the index of its block      (0 .. gridDim.x - 1)
//   blockDim.x   the number of threads per block
//   gridDim.x    the number of blocks
// and the classic GLOBAL index of a thread is blockIdx.x * blockDim.x + threadIdx.x
__global__ void hello_kernel() {
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("  hello from thread %d of block %d -> global index %d\n",
           threadIdx.x, blockIdx.x, global_index);
}


// ---------------------------------------------------------------
// PART 3 : a 2D grid of 2D blocks
// ---------------------------------------------------------------
// Blocks and grids can have 2 or 3 dimensions (type dim3), which is
// convenient for images or matrices : thread (x, y) handles element
// (row = y, col = x). Here a matrix of 6 rows x 8 columns is covered by
// blocks of 3 rows x 4 columns of threads (blockDim.y = 3, blockDim.x = 4) ;
// each thread writes the number of its block in its element, to see which
// block handles which part.
__global__ void which_block(int* owner, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {  // never forget the bounds check
        owner[row * cols + col] = blockIdx.y * gridDim.x + blockIdx.x;
    }
}


int main() {

    std::cout << "=== Part 1 : device properties ===" << std::endl;
    print_device_properties();

    std::cout << std::endl << "=== Part 2 : hello_kernel<<<2, 4>>> ===" << std::endl;
    // <<<number of blocks, threads per block>>> : 2 blocks of 4 threads.
    // A kernel launch is ASYNCHRONOUS : the host goes on immediately.
    hello_kernel<<<2, 4>>>();
    CUDA_CHECK(cudaGetLastError());       // was the launch accepted ?
    CUDA_CHECK(cudaDeviceSynchronize());  // wait for the kernel (and flush its printf)

    std::cout << std::endl << "=== Part 3 : 6 x 8 matrix, blocks of 3 x 4 threads, number of the block"
              << " of each element ===" << std::endl;
    const int rows = 6, cols = 8;
    dim3 block(4, 3);                                // 4 threads along x, 3 along y
    dim3 grid((cols + block.x - 1) / block.x,        // enough blocks to cover
              (rows + block.y - 1) / block.y);       // every element : 2 x 2

    // memory on the GPU (cudaMalloc / cudaMemcpy, see example 2)
    int* d_owner = nullptr;
    CUDA_CHECK(cudaMalloc(&d_owner, rows * cols * sizeof(int)));
    which_block<<<grid, block>>>(d_owner, rows, cols);
    CUDA_CHECK(cudaGetLastError());

    int owner[rows * cols];
    // cudaMemcpy waits for the kernel to finish before copying
    CUDA_CHECK(cudaMemcpy(owner, d_owner, sizeof(owner), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_owner));

    for (int r = 0; r < rows; ++r) {
        std::cout << "  ";
        for (int c = 0; c < cols; ++c) std::cout << owner[r * cols + c] << " ";
        std::cout << std::endl;
    }
    return 0;
}
