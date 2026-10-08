# OpenACC

OpenACC is a **directive-based** programming model for **accelerators**, GPUs most of the time. Like OpenMP, you start from a sequential C, C++ or Fortran code and annotate its loops with `#pragma acc ...`; a compiler without OpenACC support ignores them and the program remains a valid sequential program.

The difference is in the spirit. OpenACC is **descriptive**: you describe *what* is parallel, and the compiler decides *how* to map it on the hardware (a GPU, or the cores of a CPU, from the same source code). OpenMP is **prescriptive**: you tell exactly what to do. In practice, porting a code to a GPU with OpenACC comes down to three questions, which structure this folder:
 
1. **Parallelism**: which loops run on the device (`kernels`, `parallel loop`)?
2. **Data**: the GPU has its own memory — what must be copied, and when? *This is where the performance is won or lost.*
3. **Mapping**: how are the iterations spread over the levels of parallelism of the hardware (`gang`, `worker`, `vector`)?

```
   +---------+   PCIe (~16-64 GB/s)   +--------------------+
   |   CPU   | <====================> |        GPU         |
   |   RAM   |                        | VRAM (~0.5-3 TB/s) |
   +---------+                        +--------------------+
      host                                   device
```

| File | Topic | Main directives and clauses |
|---|---|---|
| [`1_OpenACC_first_steps.cpp`](1_OpenACC_first_steps.cpp) | first offloaded loops, `kernels` vs `parallel`, a reduction | `kernels`, `parallel loop`, `loop independent`, `reduction`, `copyin`, `copy` |
| [`2_OpenACC_data_management.cpp`](2_OpenACC_data_management.cpp) | keeping the data on the device | `data`, `enter data`, `exit data`, `present`, `create`, `update` |
| [`3_OpenACC_loop_parallelism.cpp`](3_OpenACC_loop_parallelism.cpp) | mapping loops on the hardware (matrix product) | `gang`, `vector`, `vector_length`, `collapse`, `tile`, `seq` |
| [`4_OpenACC_reductions_atomics.cpp`](4_OpenACC_reductions_atomics.cpp) | many threads updating shared data | `reduction`, `atomic update`, `atomic capture`, `routine` |
| [`5_OpenACC_async.cpp`](5_OpenACC_async.cpp) | asynchronous execution, overlapping transfers and computations | `async`, `wait` |
| [`6_OpenACC_heat_2D.cpp`](6_OpenACC_heat_2D.cpp) | **case study**: the 2D heat equation of the MPI folder, on a GPU | everything above |

---

## Compiling and running

```bash
make                                 # nvc++ if it is installed, g++ otherwise
make COMPILER=nvhpc TARGET=multicore # nvc++, OpenACC on the cores of the CPU
make COMPILER=gcc                    # g++ -fopenacc
make clean
```

| Compiler | Command | Runs on |
|---|---|---|
| **NVIDIA HPC SDK** (free, the reference implementation) | `nvc++ -acc -Minfo=accel -O2 file.cpp -o file` | NVIDIA GPU |
| | `nvc++ -acc=multicore -Minfo=accel -O2 file.cpp -o file` | all the cores of the CPU |
| **GCC** | `g++ -fopenacc -O2 file.cpp -o file` | NVIDIA / AMD GPU if GCC was built with offloading support (Debian/Ubuntu package `gcc-offload-nvptx`), otherwise **one** CPU thread |

- **`-Minfo=accel`**: nvc++ explains what it did with each loop. Read it every time: it tells you which loops were parallelized and how, and which data is moved. It looks like this:

  ```
  saxpy_parallel(int, float, const float *, float *):
       ..., Generating copyin(x[:n]) [if not already present]
            Generating copy(y[:n]) [if not already present]
            Generating NVIDIA GPU code
            ..., #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */
  ```

- Useful environment variables with nvc++: `NV_ACC_NOTIFY=1` prints each kernel launch, `NV_ACC_NOTIFY=2` each data transfer (`3` both), `NV_ACC_TIME=1` prints a profile when the program ends. `ACC_DEVICE_TYPE` and `ACC_DEVICE_NUM` choose the device (standard OpenACC).
- To see the timeline of kernels and transfers: `nsys profile ./program` (NVIDIA Nsight Systems).

**About the sample outputs.** They come from a laptop **without an NVIDIA GPU**, compiled with GCC 11 (`_OPENACC = 201711`, i.e. OpenACC 2.6): the compute regions run on the host, on a single thread. They show that every example is *correct*; the timings, however, say nothing about a GPU — each section explains what changes on a real one.

---

## `1_OpenACC_first_steps.cpp` — first offloaded loops

### Goal

Offload a first loop to the GPU, understand the difference between the two compute constructs, and write a reduction.

### Key concepts

**`kernels` vs `parallel loop`:**

| | `#pragma acc kernels` | `#pragma acc parallel loop` |
|---|---|---|
| who decides the loop is parallel | the **compiler**, after analysis | the **programmer**, who asserts it |
| if the compiler can't prove it is safe | the loop stays sequential | no analysis: it's parallel |
| typical use | quick first port of a region with several loops | the everyday construct |

With plain pointers, the compiler can't know whether `x` and `y` overlap (if `y == x + 1`, iteration `i` would write what iteration `i+1` reads): under `kernels`, it keeps the SAXPY loop **sequential** — on a GPU, a single thread for the whole loop — and `-Minfo=accel` reports a loop-carried dependence. Three ways to fix it: `#pragma acc loop independent` (the programmer guarantees independence), `__restrict__` pointers (they don't overlap), or `parallel loop`.

**Data clauses.** A pointer has no size: the compiler must be told how much to copy, with an *array section* `x[first:count]`:
- `copyin(x[0:n])`: host → device before the region,
- `copyout(y[0:n])`: device → host after the region,
- `copy(y[0:n])`: both.

**Reduction.** Same clause as in OpenMP: `parallel loop reduction(+:sum)`. Since OpenACC 2.7, a reduction variable of a combined construct is copied back automatically; `copy(sum)` makes it explicit for older compilers.

**Runtime API** (`#include <openacc.h>`, guarded by `#ifdef _OPENACC`): `acc_get_num_devices(acc_device_nvidia)`, `acc_get_device_type()`, and `acc_on_device(acc_device_host)` which, called *inside* a compute region, tells where the code really runs.

### What you should observe

```
compiled with OpenACC, _OPENACC = 201711
NVIDIA GPUs available : 0
compute regions run on : the HOST (no GPU, or compiled for the CPU)

=== Part 1 : saxpy (y = 2 x + y, repeated 3 times) ===
kernels                    : correct | 0.0659398 s (transfers included)
kernels + loop independent : correct | 0.0641676 s (transfers included)
parallel loop              : correct | 0.0521862 s (transfers included)

=== Part 2 : dot product (reduction) ===
dot = 1.17441e+08 (expected 1.17441e+08) | 0.0450567 s
```

On a GPU, the plain `kernels` version is dramatically slower than the two others (one thread instead of millions). Note also that each call copies 128 MB to the GPU and back: for such a light computation, the transfers take far more time than the computation itself — the subject of example 2.

### Compile and run

```bash
nvc++ -acc -Minfo=accel -O2 1_OpenACC_first_steps.cpp -o 1_OpenACC_first_steps
./1_OpenACC_first_steps
```

---

## `2_OpenACC_data_management.cpp` — keeping the data on the device

### Goal

Learn to control *when* data moves between the host and the device — the single most important skill of GPU programming.

### Key concepts

The PCIe bus between the host and the GPU is 20 to 100 times slower than the memory of the GPU. A compute region that copies its arrays in and out at every call can easily spend more than 90 % of its time in transfers.

| Clause | At the beginning of the region | At the end |
|---|---|---|
| `copyin(a[0:n])` | allocate on the device + copy host → device | free |
| `copyout(a[0:n])` | allocate | copy device → host + free |
| `copy(a[0:n])` | allocate + copy in | copy out + free |
| `create(a[0:n])` | allocate (temporary data the host never needs) | free |
| `present(a[0:n])` | check that the data is **already** on the device (runtime error otherwise) | — |

If an array is already present on the device, `copyin` / `copy` / `create` do nothing: data only moves where it *enters* the device.

The problem: smoothing a signal of 1 M values (`b[i] = (a[i-1] + a[i] + a[i+1]) / 3`), 100 iterations of 2 smoothings each, three ways:

- **A. naive**: every `parallel loop` has its own data clauses, so `a` and `b` travel to the GPU and back at every loop.
- **B. structured data region**: `#pragma acc data copy(a[0:n]) create(b[0:n])` around the iteration loop. The data lives on the device during the region; the loops inside find it `present`: **no transfer at all** during the iterations.
- **C. unstructured data**: in a real code, the data lives on the device across several functions (initialization, solver, output), which a block of code can't express. `#pragma acc enter data` and `#pragma acc exit data` can be placed anywhere, like `new` and `delete`. While the data is on the device, **`update`** synchronizes the two copies on demand: `update self(a[0:n])` (device → host, also spelled `update host`) and `update device(a[0:n])` (host → device). An update can move a part of an array only: here, a single element (`a[n/2:1]`) is read, modified on the host and sent back.

| | structured (`data`) | unstructured (`enter data` / `exit data`) |
|---|---|---|
| lifetime | a block of code | between two arbitrary points of the program |
| typical use | a solver loop inside one function | data allocated at initialization and kept for the whole run, C++ classes (constructor / destructor) |

### What you should observe

```
=== smoothing a signal of 4.1943 MB, 100 iterations ===
A. a copy at every kernel : 0.731 s | data moved : 2.517 GB
B. acc data region        : 0.707 s | data moved : 0.008 GB | same result as A : yes
C. enter / exit data + update :
    half-way, a[n/2] = 0.000 -> set to 1000 on the host
                              0.706 s | data moved : 0.008 GB + 2 floats | a[n/2] at the end = 48.769 (the spike was smoothed out)
```

On the CPU fallback nothing is actually copied, so the 3 times are equal; the volumes printed are those a GPU would have to move, computed from the data clauses. With a PCIe bus at ~25 GB/s, version A spends about 0.1 s in transfers alone, while the 200 kernels themselves take a few milliseconds: on a GPU, versions B and C are **tens of times** faster than A. Run them with `NV_ACC_NOTIFY=2` to see each transfer.

**Note**: NVIDIA GPUs can also use *unified (managed) memory*, where the driver migrates pages automatically between host and device (`nvc++ -acc -gpu=mem:managed`, or `-gpu=managed` with older versions of the SDK). It is a convenient first step, but explicit data management usually remains faster and is the only portable option.

### Compile and run

```bash
nvc++ -acc -Minfo=accel -O2 2_OpenACC_data_management.cpp -o 2_OpenACC_data_management
NV_ACC_NOTIFY=2 ./2_OpenACC_data_management
```

---

## `3_OpenACC_loop_parallelism.cpp` — mapping loops on the hardware

### Goal

Control how the iterations of nested loops are spread over the three levels of parallelism of OpenACC, on the classic matrix product `C = A × B`.

### Key concepts

| Level | Granularity | Synchronization | NVIDIA GPU | Multicore CPU |
|---|---|---|---|---|
| `gang` | coarse | none between gangs | thread block | thread |
| `worker` | medium | inside a gang | warp (32 threads) | — |
| `vector` | fine (SIMT / SIMD) | inside a worker | threads of a warp | SIMD lanes |

By default the compiler chooses the mapping itself (a `parallel loop` usually becomes `gang vector`). The loop clauses give control: `gang`, `worker`, `vector`, `seq` (run sequentially), `collapse(n)` (merge `n` nested loops), `tile(a, b)` (cut the iteration space into tiles); and on the `parallel` construct, `num_gangs`, `num_workers`, `vector_length` set the sizes. Variables declared inside a loop are private; `private(x)` privatizes a variable declared outside.

The 4 versions of the matrix product:

1. **`collapse(2)`**: the `i` and `j` loops become one loop of N² iterations, each computing one `C[i][j]` with a sequential `k` loop. Simple and often good enough.
2. **`gang` on rows, `vector` on columns**: the threads of a warp take **consecutive** `j`, so at each `k` they read consecutive `B[k][j]` and write consecutive `C[i][j]`. On a GPU, such accesses are **coalesced** into a few wide memory transactions (see [`../CUDA/4_CUDA_memory_coalescing.cu`](../CUDA/4_CUDA_memory_coalescing.cu)).
3. **the same, upside down** (`gang` on columns, `vector` on rows): consecutive threads now access elements `N` floats apart, and each one needs its own memory transaction. On a GPU this version is much slower — *the vector level must always run along the contiguous dimension of the arrays*.
4. **`tile(32, 32)`**: each gang processes a 32 × 32 tile of `C`; the rows of `A` and columns of `B` needed by a tile are reused from the cache by its 1024 elements.

### What you should observe

```
C = A x B with N = 512 (0.268435 GFLOP per product)
1. collapse(2)                 : 0.342 s |    0.79 GFLOP/s | max relative error 0.0e+00
2. gang (rows) + vector (cols) : 0.327 s |    0.82 GFLOP/s | max relative error 0.0e+00
3. gang (cols) + vector (rows) : 0.330 s |    0.81 GFLOP/s | max relative error 0.0e+00
4. tile(32, 32)                : 0.332 s |    0.81 GFLOP/s | max relative error 0.0e+00
```

On one CPU thread, all the mappings are equivalent: these are GPU questions. On a GPU, run with a bigger size (`./3_OpenACC_loop_parallelism 4096`): versions 1, 2 and 4 should be much faster than version 3. The relative error may become ~1e-6 there, because GPUs use fused multiply-add (FMA) instructions, which round differently. (For real matrix products, use a library like cuBLAS: it is another order of magnitude faster than these simple loops.)

### Compile and run

```bash
nvc++ -acc -Minfo=accel -O2 3_OpenACC_loop_parallelism.cpp -o 3_OpenACC_loop_parallelism
./3_OpenACC_loop_parallelism 4096
```

---

## `4_OpenACC_reductions_atomics.cpp` — reductions and atomics

### Goal

Thousands of GPU threads updating shared data: the race condition problem of [`../OpenMP/1_OpenMP_comparaison.cpp`](../OpenMP/1_OpenMP_comparaison.cpp), at a much larger scale, with the same two answers.

### Key concepts

- **`reduction(op:var)`** when the result is **one** value combined with an associative operation: `+ * max min & | ^ && ||`. Several reductions can appear in the same loop, and a reduction can cover a `collapse`d loop nest.
- **`atomic`** when every thread updates a shared location that depends on its data:
  - `#pragma acc atomic update`: `x++`, `x += expr`...
  - `#pragma acc atomic capture`: read the old value **and** update, as a single atomic operation (`v = x++`),
  - `#pragma acc atomic read` / `write`.
- **`#pragma acc routine seq`** compiles a function for the device too (here, the hash function that generates the data), so that it can be called inside a compute region.

### What you should observe

```
=== Part 1 : reductions ===
sum = 4991472253, min = 0, max = 999, values >= 500 : 4992523   -> correct

=== Part 2 : histogram with atomic update ===
  [0, 100) : 201279
  [100, 200) : 600983
  ...
  [900, 1000) : 198539
  -> correct

=== Part 3 : compaction with atomic capture ===
453 values >= 995 (expected 453), first indices found : 26287 58145 68398 78217 136077 139217 ...
after sorting, same indices as the sequential version : yes
```

- **Part 2**: 10 million threads and 10 bins. Without `atomic`, increments would be lost. Atomics on the GPU memory are fast, but the updates of the **same** address are still serialized: the fewer the bins, the higher the contention.
- **Part 3**: *stream compaction* — keep only the indices of the values ≥ 995 in a compact array. Each thread that finds one reserves the next free slot with `k = count++` under `atomic capture`, so that no two threads ever get the same slot. Only the `count` useful elements are brought back (`update self(selected[0:count])`). On a GPU, the slots are taken in the order the threads arrive: the **set** of indices is right, but their **order** changes from run to run (on one CPU thread they come out sorted). Sorting gives back the sequential result.

### Compile and run

```bash
nvc++ -acc -Minfo=accel -O2 4_OpenACC_reductions_atomics.cpp -o 4_OpenACC_reductions_atomics
./4_OpenACC_reductions_atomics
```

---

## `5_OpenACC_async.cpp` — asynchronous execution

### Goal

Let the host and the device work at the same time, and overlap data transfers with computations.

### Key concepts

By default, a compute region or an `update` is **synchronous**: the host waits until it is finished. The `async(q)` clause puts the operation in the **activity queue** `q` (a CUDA *stream*) and the host continues immediately:
- the operations of the **same** queue execute in order,
- the operations of **different** queues may run at the same time,
- `#pragma acc wait(q)` waits for queue `q`, `#pragma acc wait` for all of them.

**Pipelining.** A GPU has separate engines for copies and for computations. Cutting the data into chunks, each going through its own *copy in → compute → copy out* sequence in one of 3 queues, lets the copies of one chunk overlap the computation of another:

```
synchronous : [ copy in everything ][ compute everything ][ copy out everything ]

pipelined   : [in 0][comp 0][out 0]
                    [in 1][comp 1][out 1]
                          [in 2][comp 2][out 2]
                                [in 3][comp 3][out 3]     -> shorter in total
```

### What you should observe

```
running on the host (CPU fallback), n = 1048576

=== Part 1 : host and device working together ===
kernel alone                     : 0.446 s
kernel, then as much host work   : 0.904 s
both at the same time (async)    : 0.889 s

=== Part 2 : pipelining (4.194 MB in, the same out) ===
synchronous          : 0.443 s
pipelined,  4 chunks : 0.472 s | same result : yes
pipelined, 16 chunks : 0.448 s | same result : yes
```

On the CPU fallback, GCC executes the asynchronous operations synchronously: no overlap, all the times are equal. The program detects whether it runs on an accelerator and then uses a 32 times bigger problem. On a GPU:
- **Part 1**: the host works while the GPU computes, the total time drops to about the time of the kernel alone.
- **Part 2**: the pipelined versions are faster than the synchronous one, the gain being bounded by the longest of the three stages (often the transfers). For the copies to be truly asynchronous, the host memory must be *pinned* (page-locked, see [`../CUDA/7_CUDA_streams.cu`](../CUDA/7_CUDA_streams.cu)); the NVIDIA runtime handles it with internal pinned buffers.
- `nsys profile ./5_OpenACC_async` shows the transfers and kernels of the 3 queues overlapping on the timeline.

### Compile and run

```bash
nvc++ -acc -Minfo=accel -O2 5_OpenACC_async.cpp -o 5_OpenACC_async
./5_OpenACC_async
```

---

## `6_OpenACC_heat_2D.cpp` — case study: the 2D heat equation on a GPU

### Goal

Solve **exactly the same problem** as [`../MPI/7_MPI_heat_2D.cpp`](../MPI/7_MPI_heat_2D.cpp) — heat diffusion in a 512 × 512 plate, a hot disk in the middle, 5000 time steps — and compare the two approaches:

| | MPI (example 7) | OpenACC (this example) |
|---|---|---|
| the code of the update | unchanged, on each block | unchanged, on the whole plate |
| distribution of the work | explicit: blocks, 2D process grid | implicit: `parallel loop` |
| data | each process owns a block + ghost cells, halos exchanged by messages | one data region: the whole plate lives on the device |
| global diagnostics | `MPI_Reduce` between processes | `reduction` clause on the device |
| where the difficulty lies | the decomposition and the communications | the transfers between host and device |

### The recipe of a good GPU port

1. **Parallelize the loops**: `parallel loop gang` on the rows, `loop vector` on the columns, which are contiguous in memory (example 3) — the natural mapping of a 2D stencil.
2. **Keep the data on the device**: one `data` region around the whole time loop; the two arrays are copied in once, the result copied out once.
3. **Compute the diagnostics on the device**: the total heat and the maximum temperature are computed by a `reduction` on the device, so only 2 numbers come back instead of the whole plate.

Two details:
- each iteration of the time loop does **two** steps, `t → t_new` then `t_new → t` ("ping-pong"), so that the result is always in `t` without any copy or pointer swap (the data clauses refer to the host addresses of the arrays);
- `update_cell` is marked `#pragma acc routine seq`: the GPU kernels and the CPU reference call the **same** function.

### What you should observe

```
plate of 512 x 512 cells, 5000 time steps
  step  1000 : total heat = 2287200, max temperature = 99.99
  step  2000 : total heat = 2287200, max temperature = 98.94
  step  3000 : total heat = 2287200, max temperature = 95.18
  step  4000 : total heat = 2287196, max temperature = 89.72
  step  5000 : total heat = 2287162, max temperature = 83.80
OpenACC version   : 2.544 s
CPU, sequential   : 3.403 s   -> speedup 1.337
max difference    : 0.0e+00
```

followed by the same temperature map as the MPI version. The diagnostics are **identical** to the ones printed by `mpirun -np 4 ../MPI/7_MPI_heat_2D`: two programs written with different models, the same physics.

On the CPU fallback the OpenACC version is even a bit faster than the plain sequential loop (with `vector`, GCC vectorizes the inner loop). On a GPU, expect it to be one or two orders of magnitude faster than the CPU reference; the maximum difference may then be ~1e-13 instead of 0, because the GPU uses fused multiply-add instructions.

### Compile and run

```bash
nvc++ -acc -Minfo=accel -O2 6_OpenACC_heat_2D.cpp -o 6_OpenACC_heat_2D
./6_OpenACC_heat_2D
```

---

## Cheat sheet

| Directive / clause | Purpose |
|---|---|
| `#pragma acc parallel loop` | run a loop on the device, the programmer asserts it is parallel |
| `#pragma acc kernels` | let the compiler find the parallel loops of a region |
| `#pragma acc serial` | run a block on the device with a single thread |
| `loop gang / worker / vector / seq` | choose the level of parallelism of a loop |
| `collapse(n)`, `tile(a, b)`, `independent`, `private(x)` | loop transformations and properties |
| `reduction(op:x)` | combine private copies at the end |
| `copyin`, `copyout`, `copy`, `create`, `present` | data clauses |
| `#pragma acc data` | structured data region |
| `#pragma acc enter data` / `exit data` (`delete`) | unstructured data lifetime |
| `#pragma acc update self(...)` / `device(...)` | synchronize host and device copies |
| `#pragma acc atomic update / capture / read / write` | atomic operations |
| `#pragma acc routine seq` | compile a function for the device |
| `async(q)`, `#pragma acc wait(q)` | asynchronous execution |

| Runtime function | Returns |
|---|---|
| `acc_get_num_devices(type)` | number of devices of a type (`acc_device_nvidia`...) |
| `acc_get_device_type()` | type of the current device |
| `acc_on_device(type)` | inside a compute region: is the code running on this type of device? |
| `acc_is_present(ptr, bytes)` | is this data present on the device? |

### OpenACC and OpenMP offloading side by side

The same concepts exist in OpenMP (see [`../OpenMP/8_OpenMP_gpu_offload.cpp`](../OpenMP/8_OpenMP_gpu_offload.cpp)):

| OpenACC | OpenMP |
|---|---|
| `parallel loop` | `target teams distribute parallel for` |
| `gang` / `worker` / `vector` | `teams` / `parallel` / `simd` |
| `data copyin / copyout / copy / create` | `target data map(to / from / tofrom / alloc)` |
| `enter data` / `exit data` | `target enter data` / `target exit data` |
| `update self` / `update device` | `target update from` / `target update to` |
| `routine seq` | `declare target` |
| `async(q)` / `wait` | `nowait` + `depend` / `taskwait` |

---

## Going further

- Profile the examples with `NV_ACC_TIME=1` and `nsys profile`, and check that the data transfers you see are the ones you expected.
- In `6_OpenACC_heat_2D.cpp`, replace the explicit mapping by `collapse(2)` or `tile(32, 32)`, try `vector_length(256)`, and compare on a GPU.
- Combine OpenACC with MPI: one MPI process per GPU, each working on a block of the plate with halos exchanged between GPUs (with a CUDA-aware MPI, `#pragma acc host_data use_device(...)` passes device addresses directly to MPI).
- Compile the same examples with `-acc=multicore` and compare with the OpenMP versions.

## References

- OpenACC specification and resources (including the *OpenACC Programming and Best Practices Guide*): <https://www.openacc.org/>
- NVIDIA HPC SDK documentation: <https://docs.nvidia.com/hpc-sdk/>
- GCC and OpenACC: <https://gcc.gnu.org/wiki/OpenACC>
- S. Chandrasekaran, G. Juckeland (eds.), *OpenACC for Programmers: Concepts and Strategies*, Addison-Wesley, 2017.
