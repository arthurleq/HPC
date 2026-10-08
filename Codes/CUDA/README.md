# CUDA

CUDA (*Compute Unified Device Architecture*) is NVIDIA's platform to use its GPUs for general-purpose computing. Unlike the directive-based models (OpenMP offloading, OpenACC), **nothing is automatic**: the programmer writes the functions that run on the GPU — the **kernels** —, chooses how many threads execute them, and manages the GPU memory and the transfers by hand. More work, but full control: most GPU libraries (cuBLAS, cuFFT, cuDNN...) and many HPC codes are written in CUDA, and AMD's HIP is an almost line-by-line copy of it.

CUDA uses the **SIMT** model (*Single Instruction, Multiple Threads*), an extension of SIMD in Flynn's taxonomy. A kernel is executed by a **grid** of **blocks** of **threads**:
 
```
   grid  (all the threads of one kernel launch)
   +--------------------+--------------------+--------------------+
   | block 0            | block 1            | block 2            |   ...
   | t0 t1 t2 ... t255  | t0 t1 t2 ... t255  | t0 t1 t2 ... t255  |
   +--------------------+--------------------+--------------------+
```

- the threads of a block run on the same **multiprocessor** (SM), can share a fast on-chip memory (*shared memory*) and synchronize with each other;
- the blocks are independent: they can run in any order, on any SM — which lets the same code scale from a laptop GPU to a data-center one;
- the threads are executed by groups of 32, the **warps**: the 32 threads of a warp execute the same instruction at the same time.

Each thread computes its position from built-in variables, the classic global index being `blockIdx.x * blockDim.x + threadIdx.x`.

| File | Topic | Main notions |
|---|---|---|
| [`1_CUDA_hello_world.cu`](1_CUDA_hello_world.cu) | device properties, first kernels, thread hierarchy | `__global__`, `<<<blocks, threads>>>`, `threadIdx`, `blockIdx`, `dim3` |
| [`2_CUDA_vector_add.cu`](2_CUDA_vector_add.cu) | the complete life cycle of a CUDA program | `cudaMalloc`, `cudaMemcpy`, `cudaFree`, events, grid-stride loop |
| [`3_CUDA_unified_memory.cu`](3_CUDA_unified_memory.cu) | one pointer for host and device | `cudaMallocManaged`, page faults, `cudaMemPrefetchAsync` |
| [`4_CUDA_memory_coalescing.cu`](4_CUDA_memory_coalescing.cu) | memory access patterns (matrix transpose) | coalescing, `__shared__`, `__syncthreads`, bank conflicts |
| [`5_CUDA_shared_memory_matmul.cu`](5_CUDA_shared_memory_matmul.cu) | data reuse with shared memory (matrix product) | tiling |
| [`6_CUDA_reduction.cu`](6_CUDA_reduction.cu) | combining millions of values into one | `atomicAdd`, tree reduction, `__shfl_down_sync`, CUB |
| [`7_CUDA_streams.cu`](7_CUDA_streams.cu) | overlapping transfers and computations | pinned memory, streams, `cudaMemcpyAsync` |
| [`8_CUDA_thrust.cu`](8_CUDA_thrust.cu) | the "STL of CUDA" | `thrust::device_vector`, `transform`, `reduce`, `sort` |

---

## Compiling and running

```bash
make                     # compiles for the GPU of this machine (-arch=native)
make CUDA_ARCH=sm_80     # ... or for a given architecture (here an A100)
make 6_CUDA_reduction    # a single example
make clean
```

or by hand: `nvcc -O3 -arch=native 2_CUDA_vector_add.cu -o 2_CUDA_vector_add`.

- `nvcc` compiles the host code with the usual C++ compiler and the device code for the GPU. `-arch` gives the *compute capability* (the GPU generation) to compile for: `-arch=native` uses the GPU of the machine; on a cluster login node (often without GPU), give it explicitly:

| GPU | Compute capability | `-arch` |
|---|---|---|
| V100 | 7.0 | `sm_70` |
| A100 | 8.0 | `sm_80` |
| RTX 30xx, A40 | 8.6 | `sm_86` |
| RTX 40xx, L40 | 8.9 | `sm_89` |
| H100, H200 | 9.0 | `sm_90` |
| B200 | 10.0 | `sm_100` |

- **Always check errors.** Every CUDA function returns a `cudaError_t`, and kernel launches return nothing: their errors are retrieved with `cudaGetLastError()` (launch errors) and at the next synchronization (errors during the execution). Every example uses the same `CUDA_CHECK` macro, which prints the error, the file and the line, then stops the program. Without it, a failed kernel goes unnoticed and the program silently prints garbage.
- **Tools** of the CUDA toolkit worth knowing from day one:
  - `compute-sanitizer ./program`: detects out-of-bounds accesses and other memory errors in kernels (`--tool racecheck` for shared memory races);
  - `nsys profile ./program` (Nsight Systems): timeline of kernels, transfers and streams;
  - `ncu ./program` (Nsight Compute): detailed metrics of each kernel (memory throughput, occupancy, bank conflicts...).

**About the outputs shown below.** These examples were written on a machine without an NVIDIA GPU: they compile with `nvcc` 12.6 for every architecture from `sm_70` to `sm_90`, and their kernels were executed in a CPU emulation to validate their logic, but no GPU timing could be measured. The outputs shown are those that do not depend on the GPU; for timings, each section describes the expected trends, which depend a lot on the GPU model.

---

## `1_CUDA_hello_world.cu` — the thread hierarchy

### Goal

Find out what GPU is available, launch a first kernel, and see how threads, blocks and grids are organized.

### Key concepts

- **Function qualifiers**: `__global__` (a kernel: called from the host, runs on the device), `__device__` (called from the device only), `__host__` (the default, host code).
- **Launch**: `kernel<<<blocks, threads_per_block>>>(arguments)`. The launch is **asynchronous**: the host goes on immediately. `cudaDeviceSynchronize()` waits for the kernel; `printf` works inside kernels, its output being flushed at synchronization points.
- **Built-in variables**: `threadIdx` (index in the block), `blockIdx` (index of the block), `blockDim` (threads per block), `gridDim` (blocks in the grid), each with `.x`, `.y`, `.z`.
- **2D/3D grids**: blocks and grids can be declared with `dim3` to match images or matrices: thread `(x, y)` handles element `(row = y, col = x)`. The number of blocks is rounded up, so threads outside of the data must do nothing: **never forget the bounds check**.
- **Device properties**: number of multiprocessors (SM), warp size (32), maximum threads per block (1024) and per SM, shared memory per block, global memory, and the theoretical bandwidth `2 × memory clock × bus width` (two transfers per clock).

### What you should observe

The properties of your GPU, then the 8 threads of `hello_kernel<<<2, 4>>>` (the order of the blocks may change from one run to the other):

```
=== Part 2 : hello_kernel<<<2, 4>>> ===
  hello from thread 0 of block 0 -> global index 0
  hello from thread 1 of block 0 -> global index 1
  hello from thread 2 of block 0 -> global index 2
  hello from thread 3 of block 0 -> global index 3
  hello from thread 0 of block 1 -> global index 4
  hello from thread 1 of block 1 -> global index 5
  hello from thread 2 of block 1 -> global index 6
  hello from thread 3 of block 1 -> global index 7
```

and which of the 2 × 2 blocks of 3 × 4 threads covers each element of a 6 × 8 matrix:

```
  0 0 0 0 1 1 1 1
  0 0 0 0 1 1 1 1
  0 0 0 0 1 1 1 1
  2 2 2 2 3 3 3 3
  2 2 2 2 3 3 3 3
  2 2 2 2 3 3 3 3
```

### Compile and run

```bash
nvcc -O3 -arch=native 1_CUDA_hello_world.cu -o 1_CUDA_hello_world
./1_CUDA_hello_world
```

---

## `2_CUDA_vector_add.cu` — the life cycle of a CUDA program

### Goal

Go through the 6 steps of every CUDA program on the simplest example, `c = a + b` for vectors of 32 M floats, and measure where the time goes.

### Key concepts

```
 1. host data             std::vector<float> a(n), b(n), c(n)
 2. device memory         cudaMalloc(&d_a, bytes)
 3. inputs  host -> GPU   cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice)
 4. kernel                vector_add<<<blocks, 256>>>(d_a, d_b, d_c, n)
 5. result  GPU -> host   cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost)
 6. free                  cudaFree(d_a)
```

- The pointers returned by `cudaMalloc` are addresses in the GPU memory: the host must never dereference them (convention: prefix them with `d_`).
- **Block size**: a multiple of the warp size (32), usually between 128 and 512. **Number of blocks**: `(n + threads - 1) / threads`, rounded up, hence the `if (i < n)` in the kernel.
- **Grid-stride loop**: `for (i = global index; i < n; i += total number of threads)` lets a grid of any size (here a few blocks per SM) process data of any size.
- **CUDA events** (`cudaEventRecord`, `cudaEventElapsedTime`) measure time on the GPU side, which is the right way to time asynchronous operations.

### What you should observe

`correct` for both kernels, and three timings that tell the main story of GPU computing:
- the **kernel** reaches an effective bandwidth (3 arrays read or written) close to the memory bandwidth of the GPU, hundreds of GB/s to a few TB/s;
- the **copies** go through the PCIe bus, at 10-25 GB/s (even less from pageable memory, see example 7);
- so the copies take **tens of times longer** than the kernel. Moving data to the GPU for such a light operation is never worth it: a GPU code must keep its data on the GPU for as long as possible and do a lot of work on it.

### Compile and run

```bash
nvcc -O3 -arch=native 2_CUDA_vector_add.cu -o 2_CUDA_vector_add
./2_CUDA_vector_add
```

---

## `3_CUDA_unified_memory.cu` — unified memory

### Goal

Use one pointer for both the host and the device, understand what it costs, and how to make it fast.

### Key concepts

- `cudaMallocManaged(&x, bytes)` returns a pointer valid on the host **and** on the device: no more `cudaMemcpy`. The driver migrates the memory **pages** on demand: when the GPU touches a page that is in the host memory, a **page fault** occurs and the page migrates (and the other way around).
- Very convenient to port a code quickly or for complex data structures (lists, trees...), but page faults are expensive.
- `cudaMemPrefetchAsync` migrates a whole range **before** it is needed, in large chunks. Its signature changed in CUDA 13 (the destination became a `cudaMemLocation`): the small `prefetch` function of the example works with both versions.
- Kernel launches remain asynchronous: the host must synchronize before reading data written by a kernel.
- On-demand migration needs a Pascal (`sm_60`) or newer GPU on Linux (`cudaDevAttrConcurrentManagedAccess`).

### What you should observe

- the **first** kernel is much slower than the second one: the pages migrate through page faults *during* the kernel;
- the **second** kernel runs at full speed: the pages are already in the GPU memory;
- with **prefetching**, the migration runs at the PCIe bandwidth and the kernel is as fast as the second one;
- `result : correct` everywhere.

### Compile and run

```bash
nvcc -O3 -arch=native 3_CUDA_unified_memory.cu -o 3_CUDA_unified_memory
./3_CUDA_unified_memory
```

---

## `4_CUDA_memory_coalescing.cu` — memory access patterns

### Goal

Most kernels are limited by the memory bandwidth: *how* threads access the memory matters as much as *how much* they access. The matrix transpose shows it, and introduces the shared memory.

### Key concepts

**Coalescing.** The 32 threads of a warp execute the same load at the same time. If they read 32 consecutive floats (128 bytes), the hardware merges them into a few wide transactions; if their addresses are far apart (*strided*), each thread needs its own 32-byte transaction to use 4 bytes of it:

```
   coalesced : thread  0  1  2  3 ... 31          strided : thread  0     1     2   ...
               address 0  4  8 12 ... 124                   address 0  16384 32768 ...
               -> 4 transactions of 32 bytes                -> 32 transactions
```

The transpose `out[j][i] = in[i][j]` reads rows (contiguous) and writes columns (strided): one of the two accesses is necessarily bad...

**Shared memory** to the rescue: a small memory on the chip (tens to a few hundred KB per SM), much faster than the global memory, shared by the threads of a block and managed by hand (`__shared__`). Each block reads a 32 × 32 tile row by row (coalesced) into shared memory, then writes it row by row (coalesced) at its transposed position: the transposition itself happens in shared memory. `__syncthreads()` is a barrier for the threads of a block: the tile must be complete before anyone reads it, since each thread writes elements loaded by other threads.

**Bank conflicts.** The shared memory is made of 32 *banks* (consecutive 4-byte words in consecutive banks). Threads of a warp accessing the same bank are serialized. Reading a **column** of `tile[32][32]` means reading words 32 apart: all in the same bank, a 32-way conflict. One padding column, `tile[32][33]`, shifts each row by one bank:

```
   tile[32][32] : column 0 = words 0, 32, 64, ...  -> banks 0, 0, 0, ...   (conflict)
   tile[32][33] : column 0 = words 0, 33, 66, ...  -> banks 0, 1, 2, ...   (no conflict)
```

### What you should observe

The 4 kernels are `correct`, and their effective bandwidths are ordered: **copy > shared memory without bank conflicts ≥ shared memory > naive transpose**. The naive transpose only reaches a fraction of the copy bandwidth; the padded version comes close to it. `ncu` shows the difference in the "global memory efficiency" and "shared memory bank conflicts" metrics.

### Compile and run

```bash
nvcc -O3 -arch=native 4_CUDA_memory_coalescing.cu -o 4_CUDA_memory_coalescing
./4_CUDA_memory_coalescing
```

---

## `5_CUDA_shared_memory_matmul.cu` — data reuse with shared memory

### Goal

Turn a memory-bound kernel into a (more) compute-bound one, on the matrix product `C = A × B` (2048 × 2048).

### Key concepts

- **Naive kernel**: one thread per element of `C`, reading its row of `A` and its column of `B` from the global memory. The same elements are read again and again by different threads (row `i` of `A` is read by all the `n` threads of row `i` of `C`): the kernel is limited by the memory traffic.
- **Tiled kernel**: a block of 16 × 16 threads computes a 16 × 16 tile of `C`. The dot products are cut into chunks of 16; for each chunk, the block loads **one** tile of `A` and **one** tile of `B` into shared memory (each thread loads one element of each), then every thread reads them 16 times from the fast shared memory: **16 times fewer** loads from the global memory.

```
                 B tile t
                   | |
     A tile t -> [C tile]        C tile = sum over t of (A tile t) x (B tile t)
```

- **Two barriers per iteration**: the first one so that the tiles are complete before being used, the second one so that nobody overwrites them while others still use them. `__syncthreads()` must be reached by **all** the threads of the block: never inside a condition that only some threads satisfy (the bounds checks are on the loads, not around the barriers). Elements outside of the matrices are loaded as 0, so any `n` works.

### What you should observe

Both kernels are `correct` (1000 elements are checked against a double-precision computation on the host, relative error around 1e-6), and the tiled kernel is several times faster than the naive one. Both remain far from the peak of the GPU: real libraries (`cublasSgemm` in cuBLAS) add register tiling, vectorized loads and tensor cores, and are another order of magnitude faster. In a real code, **use cuBLAS**.

### Compile and run

```bash
nvcc -O3 -arch=native 5_CUDA_shared_memory_matmul.cu -o 5_CUDA_shared_memory_matmul
./5_CUDA_shared_memory_matmul
```

---

## `6_CUDA_reduction.cu` — parallel reduction

### Goal

Sum 64 M integers: trivial sequentially, it is *the* classic exercise of GPU programming, because millions of threads must combine their values into one.

### The 4 versions

**1. One `atomicAdd` per element** on a single global counter. Correct, but 64 M atomic operations on the **same** address are serialized.

**2. Tree reduction in shared memory.** Each block loads 256 values into shared memory and halves them at each step, then does **one** `atomicAdd`:

```
   256 values -> 128 partial sums -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1      (8 steps)
   step s : cache[t] += cache[t + s]  for t < s,  then __syncthreads()
```

The active threads are contiguous (`t < s`): whole warps work or rest together, and there is no bank conflict.

**3. Warp shuffles + grid-stride loop.** `__shfl_down_sync(mask, v, offset)` reads the value of `v` in the thread `offset` lanes further **in the same warp**, directly from its registers: no shared memory, no `__syncthreads()`. After 5 steps (offsets 16, 8, 4, 2, 1), lane 0 holds the sum of its warp. Moreover, each thread first sums many elements alone (grid-stride loop): only a few blocks, a few atomics, more work per thread.

**4. A library**: `cub::DeviceReduce::Sum`. CUB (shipped with CUDA) provides tuned reductions, scans, sorts... In real codes, use the libraries.

### What you should observe

The 4 versions give the same, exact sum (`sum = 301989882  correct`: integer sums don't depend on the order of the additions). Version 1 is by far the slowest; version 2 is much faster; versions 3 and 4 get close to the memory bandwidth of the GPU (the sum only reads each value once, so it is memory-bound).

### Compile and run

```bash
nvcc -O3 -arch=native 6_CUDA_reduction.cu -o 6_CUDA_reduction
./6_CUDA_reduction
```

---

## `7_CUDA_streams.cu` — overlapping transfers and computations

### Goal

Hide the cost of the transfers by overlapping them with the computations: the same idea as [`../OpenACC/5_OpenACC_async.cpp`](../OpenACC/5_OpenACC_async.cpp), written by hand.

### Key concepts

**Pinned memory.** Memory from `new` / `malloc` is *pageable*: the OS may move its pages. The copy engine of the GPU (DMA) can only read page-locked memory, so the driver first copies pageable data into an internal pinned buffer, then transfers it. `cudaMallocHost` allocates **pinned** memory directly: faster copies, and **required** for truly asynchronous ones. It can't be swapped out: don't pin gigabytes of it.

**Streams.** A stream is a queue of GPU operations executed in order; operations of different streams may run concurrently. A GPU has separate engines for host → device copies, device → host copies and kernels: cutting the data into chunks, each handled by a stream (`cudaMemcpyAsync` + kernel launched with `<<<blocks, threads, 0, stream>>>` + `cudaMemcpyAsync`), keeps all of them busy:

```
   1 stream  : [ H->D everything ][ kernel on everything ][ D->H everything ]

   4 streams : [H->D 0][kernel 0][D->H 0]
                       [H->D 1][kernel 1][D->H 1]
                               [H->D 2][kernel 2][D->H 2]
                                       [H->D 3][kernel 3][D->H 3]
```

### What you should observe

- **Part 1**: copies from pinned memory are faster than from pageable memory.
- **Part 2**: the versions with 4 streams are faster than the single-stream one, with `same result : yes`; at best, the total time tends to the longest of the three stages (often the transfers). `nsys profile ./7_CUDA_streams` shows the copies and kernels of the 4 streams overlapping on the timeline.

### Compile and run

```bash
nvcc -O3 -arch=native 7_CUDA_streams.cu -o 7_CUDA_streams
./7_CUDA_streams
```

---

## `8_CUDA_thrust.cu` — Thrust, the "STL of CUDA"

### Goal

Use the GPU without writing a single kernel.

### Key concepts

Thrust is a header-only C++ library shipped with CUDA, modeled on the STL:
- **containers**: `thrust::device_vector<T>` lives in the GPU memory, allocated and freed automatically (RAII), and copied with `=` (`host_vector h = d;` is a device → host copy);
- **algorithms**: `transform`, `reduce`, `transform_reduce`, `count_if`, `minmax_element`, `sort`, `inclusive_scan`... run on the GPU when given device iterators;
- the operations applied to the elements are **functors**, structures whose `operator()` is `__host__ __device__` so that the GPU can call them;
- **fancy iterators** like `counting_iterator` generate sequences on the fly, without storing them;
- errors are reported as C++ exceptions.

Example 6 needed four hand-written kernels to sum an array; with Thrust, it is one line: `thrust::reduce(d.begin(), d.end())`.

### What you should observe

The statistics of 16 M pseudo-random values (deterministic, so identical on every machine):

```
=== statistics of 16777216 uniform random values in [0, 1) ===
mean     = 0.50006   (expected 0.5)
std dev  = 0.28867   (expected 1/sqrt(12) = 0.28868)
min, max = 0, 0.99999994
> 0.9    : 1677537 values (expected ~1677721)
```

then the sorting of the same 16 M floats with `std::sort` on the CPU and `thrust::sort` on the GPU, with `same result : yes` — the GPU sort being typically tens of times faster.

### Compile and run

```bash
nvcc -O3 -arch=native 8_CUDA_thrust.cu -o 8_CUDA_thrust
./8_CUDA_thrust
```

---

## The memory hierarchy of a GPU

| Memory | Visible by | Size (orders of magnitude) | Speed |
|---|---|---|---|
| registers | one thread | up to 255 × 32-bit per thread | fastest |
| shared memory | the threads of a block | tens to a few hundred KB per SM | close to registers (beware of bank conflicts) |
| L1 / L2 caches | an SM / the whole GPU | ~100-250 KB per SM / tens of MB | automatic |
| global memory (HBM, GDDR) | all the threads, the host through copies | 8 to 200 GB | 0.5 to 8 TB/s, high latency |
| host memory | the host, the GPU through PCIe / NVLink | up to TBs | 16-64 GB/s (PCIe) |

Most of GPU optimization consists in moving the data up this hierarchy: few transfers to the GPU (examples 2, 3, 7), coalesced accesses to the global memory (example 4), data reused from shared memory or registers (examples 5, 6).

## Cheat sheet

| Notion | Syntax |
|---|---|
| kernel / device function | `__global__ void k(...)` / `__device__ T f(...)` |
| launch | `k<<<blocks, threads, shared_bytes, stream>>>(args)` |
| thread position | `threadIdx`, `blockIdx`, `blockDim`, `gridDim` (`.x .y .z`), `dim3` |
| device memory | `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemset` |
| unified / pinned memory | `cudaMallocManaged`, `cudaMemPrefetchAsync` / `cudaMallocHost`, `cudaFreeHost` |
| synchronization | `cudaDeviceSynchronize()` (host), `__syncthreads()` (block), `__syncwarp()` (warp) |
| shared memory | `__shared__ float tile[32][33];` |
| atomics | `atomicAdd`, `atomicMax`, `atomicCAS`... |
| warp primitives | `__shfl_down_sync`, `__shfl_sync`, `__ballot_sync` |
| streams and events | `cudaStreamCreate`, `cudaMemcpyAsync`, `cudaEventRecord`, `cudaEventElapsedTime` |
| errors | `cudaGetLastError`, `cudaGetErrorString` |

| CUDA | OpenACC | OpenMP offloading |
|---|---|---|
| kernel launch | `parallel loop` | `target teams distribute parallel for` |
| block | gang | team |
| warp / thread | worker / vector | thread / simd lane |
| `cudaMemcpy` | `copyin`, `copyout`, `update` | `map`, `target update` |
| stream | `async(q)` queue | `nowait` + `depend` |

---

## Going further

- Run every example under `compute-sanitizer`, then break one on purpose (remove the bounds check of `vector_add`, or a `__syncthreads()` of `matmul_tiled`) and see the errors it reports.
- Occupancy: `cudaOccupancyMaxPotentialBlockSize` suggests a block size, and `ncu` tells how many warps are active per SM; try block sizes from 32 to 1024 in example 2.
- In example 5, use 32 × 32 tiles, then compare with `cublasSgemm` (link with `-lcublas`).
- CUDA graphs (launching a whole sequence of kernels at once), cooperative groups, tensor cores (`wmma` / cuBLAS), multi-GPU programming (`cudaSetDevice`, NCCL), and CUDA-aware MPI to send GPU buffers directly between nodes.
- Porting to AMD GPUs: HIP has almost the same API (`hipMalloc`, `hipMemcpy`, same kernel syntax), and `hipify` converts CUDA code automatically.

## References

- CUDA C++ Programming Guide: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- CUDA C++ Best Practices Guide: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- Thrust and CUB (CUDA Core Compute Libraries): <https://nvidia.github.io/cccl/>
- M. Harris, *Optimizing Parallel Reduction in CUDA* (NVIDIA), and *An Efficient Matrix Transpose in CUDA C/C++* (NVIDIA Technical Blog): the classic studies behind examples 4 and 6.
- W. Hwu, D. Kirk, I. El Hajj, *Programming Massively Parallel Processors*, 4th edition, Morgan Kaufmann, 2022.
