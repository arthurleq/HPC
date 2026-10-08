# OpenMP

OpenMP is the programming model associated with **shared memory**: an API (compiler directives, a small runtime library and a few environment variables) to program this kind of architecture in C, C++ and Fortran.
The principle is to create several **threads** that work in parallel inside a single process, all of them seeing the same memory.
For example, when working on a loop, its iterations can be shared among several threads.

Its main strength is to be **incremental**: you start from a working sequential code and add `#pragma omp ...` directives where the time is spent. A compiler without OpenMP support simply ignores the pragmas, and the code remains a valid sequential program.

The goal of this folder is to explain how OpenMP works in practice: each C++ file below illustrates one aspect of OpenMP with heavily commented code, and this README explains the concepts, what you should observe when running each program, and how to compile it. The theory behind it (Flynn's taxonomy, memory architectures, NUMA, Amdahl's law) is covered in [`HPC_fr.pdf`](../../HPC_fr.pdf) (in French).

| File | Topic | Main constructs |
|---|---|---|
| [`0_OpenMP_hello_world.cpp`](0_OpenMP_hello_world.cpp) | fork-join model, number of threads, first parallel loop and its speedup | `parallel`, `parallel for`, `num_threads`, `if` |
| [`1_OpenMP_comparaison.cpp`](1_OpenMP_comparaison.cpp) | 4 ways to update a shared variable | `reduction`, `atomic`, `critical` |
| [`2_OpenMP_scheduling.cpp`](2_OpenMP_scheduling.cpp) | sharing an irregular loop among threads | `schedule(static / dynamic / guided)` |
| [`3_OpenMP_numa_affinity.cpp`](3_OpenMP_numa_affinity.cpp) | where threads run, where memory lives | `proc_bind`, `OMP_PLACES`, first touch |
| [`4_OpenMP_data_sharing.cpp`](4_OpenMP_data_sharing.cpp) | shared vs private variables, a classic data race | `private`, `firstprivate`, `lastprivate`, `default(none)` |
| [`5_OpenMP_worksharing_sync.cpp`](5_OpenMP_worksharing_sync.cpp) | other work-sharing and synchronization constructs | `single`, `master`, `barrier`, `nowait`, `sections`, `ordered`, locks |
| [`6_OpenMP_tasks.cpp`](6_OpenMP_tasks.cpp) | recursive and irregular parallelism | `task`, `taskwait`, `depend` |
| [`7_OpenMP_simd_collapse.cpp`](7_OpenMP_simd_collapse.cpp) | vectorization, nested loops | `simd`, `declare simd`, `parallel for simd`, `collapse` |
| [`8_OpenMP_gpu_offload.cpp`](8_OpenMP_gpu_offload.cpp) | offloading computations to a GPU | `target`, `teams`, `distribute`, `map`, `target data` |

The files are numbered in a suggested reading order. If you are new to OpenMP, reading `4` (data sharing) right after `0` is a good idea: most OpenMP bugs come from there.

---

## Compiling and running

```bash
make                       # builds every example of the folder
make 2_OpenMP_scheduling   # builds only one of them
make clean                 # removes the executables
```

or by hand, for any file:

```bash
g++ -fopenmp -O2 2_OpenMP_scheduling.cpp -o 2_OpenMP_scheduling
./2_OpenMP_scheduling
```

- `-fopenmp` turns OpenMP on with GCC and Clang (`-qopenmp` with Intel `icpx`, `-mp` with NVIDIA `nvc++`). Without it, the pragmas are ignored and the program is sequential.
- The behaviour of the runtime is controlled by environment variables:

| Variable | Role | Example |
|---|---|---|
| `OMP_NUM_THREADS` | number of threads of a parallel region | `OMP_NUM_THREADS=4 ./0_OpenMP_hello_world` |
| `OMP_SCHEDULE` | schedule used by `schedule(runtime)` loops | `OMP_SCHEDULE="dynamic,64"` |
| `OMP_PLACES` | where threads may run (`threads`, `cores`, `sockets`...) | `OMP_PLACES=cores` |
| `OMP_PROC_BIND` | how threads are bound to the places (`close`, `spread`, `true`...) | `OMP_PROC_BIND=spread` |
| `OMP_DISPLAY_ENV` | prints the whole configuration of the runtime at startup | `OMP_DISPLAY_ENV=true` |

The sample outputs shown below come from a laptop with an Intel Core i7-1185G7 (**4 cores, 8 hardware threads**, a single NUMA node), compiled with GCC 11. Timings on a laptop are noisy (other programs, turbo frequency...) and will be different on your machine: what matters is the **trend**, not the exact numbers.

---

## `0_OpenMP_hello_world.cpp` — the fork-join model

### Goal

See threads appear and disappear, learn how to choose how many of them there are, and measure the speedup of a first parallel loop.

### Key concepts

**Fork-join.** A program always starts with a single thread, the *master* (or *initial*) thread. When it reaches `#pragma omp parallel`, it **forks** a team of threads that all execute the following block. At the end of the block, every thread waits for the others (*implicit barrier*) and the team **joins** back into the master thread alone:

```
                 fork                        join
   master ---------+---> thread 0 ------------+---> master
                   +---> thread 1 ------------+
                   +---> thread 2 ------------+
                   +---> thread 3 ------------+
```

Inside the region, each thread can ask for its identity: `omp_get_thread_num()` returns its number (from 0 to N-1) and `omp_get_num_threads()` the size of the team (1 outside of a parallel region). The output lines appear in a **different order at each run**: the threads run concurrently and nothing decides who is "first". The `critical` around `std::cout` only prevents two lines from mixing their characters.

**Choosing the number of threads.** From the lowest to the highest priority:

| Mechanism | Scope |
|---|---|
| `OMP_NUM_THREADS=4 ./program` | the whole program, without recompiling |
| `omp_set_num_threads(4);` | the following parallel regions |
| `#pragma omp parallel num_threads(4)` | this region only |

By default the runtime uses one thread per logical core (`omp_get_num_procs()`, hyper-threads included). The `if(condition)` clause runs the region with a team only when the condition is true: forking a team costs a few microseconds, which is not worth it for a tiny problem.

**The `_OPENMP` macro** is only defined when compiling with OpenMP. Its value is the release date of the supported specification: `201511` = OpenMP 4.5 (GCC 11), `201811` = 5.0, `202011` = 5.1, `202111` = 5.2, `202411` = 6.0.

**A first parallel loop.** `#pragma omp parallel for` creates a team **and** shares the iterations of the loop among its threads (each thread gets about `n / P` iterations). It is only correct because the iterations are **independent**: iteration `i` only writes `y[i]` and only reads `x[i]`. The loop index is automatically private to each thread, the arrays declared outside are shared.

**Speedup and efficiency.** With `T(P)` the time on `P` threads, the speedup is `S(P) = T(1) / T(P)` and the efficiency `E(P) = S(P) / P` (1 = perfect). Amdahl's law reminds us that the sequential part of a program limits the speedup, whatever the number of threads.

### What you should observe

```
threads | time (s) | speedup S = T(1)/T(P) | efficiency S/P
    1   |    0.352 |           1.000       |      1.000
    2   |    0.186 |           1.895       |      0.948
    4   |    0.097 |           3.639       |      0.910
    8   |    0.152 |           2.318       |      0.290
```

- The speedup is almost linear up to the number of **physical cores** (4 here): the loop body is deliberately expensive (`sin`, `exp`, `sqrt`), so it is limited by computation, not by memory.
- Going from 4 to 8 threads brings nothing, or even slows down: the two hyper-threads of a core share the same execution units, which this loop already keeps busy, and the other programs of the machine now compete for the cores.
- The results are identical for every number of threads: the work is split differently, but each `y[i]` is computed in exactly the same way.

The timing function keeps the best of 3 runs, and the arrays are written once before any measurement: the very first write to freshly allocated memory triggers *page faults* (the OS only maps a page when it is first touched), which would unfairly slow down the first measurement.

### Compile and run

```bash
g++ -fopenmp -O2 0_OpenMP_hello_world.cpp -o 0_OpenMP_hello_world
./0_OpenMP_hello_world
OMP_NUM_THREADS=2 ./0_OpenMP_hello_world
```

---

## `1_OpenMP_comparaison.cpp` — memory synchronization strategies

### Goal

Compute the sum of the elements of an array of 10 million integers, in parallel, with four different strategies to protect the shared variable `sum` — and see which one is the fastest.

### The 4 functions

| Function | OpenMP clause | When the synchronization happens | Cost |
|---|---|---|---|
| `sum_table_unprotected` | none | never | **wrong** result |
| `sum_table_reduction` | `reduction(+:sum)` | only once, at the end | almost zero |
| `sum_table_atomic` | `atomic` | at every iteration (hardware instruction) | small but not zero |
| `sum_table_critical` | `critical` | at every iteration (software lock) | high |

**`sum_table_unprotected`** — several threads execute `sum += table[i]` at the same time on the same variable. This is a *race condition*: `+=` is not an atomic operation (it is made of a read, an addition and a write), so updates are lost when two threads read the same value of `sum` before one of them has finished writing it. The result is wrong and changes from one run to the other — this function is only there to illustrate the problem, never use it in practice.

**`sum_table_reduction`** — each thread gets a private copy of `sum` (initialized to 0, the neutral element of `+`), accumulates into it without ever touching the copies of the other threads, then OpenMP merges all the copies in one go at the end of the loop. Since there is no synchronization at all *during* the loop, it is almost free: this is the solution to prefer whenever the operation allows it (sum, product, min, max, logical and, or...).

**`sum_table_atomic`** — uses a dedicated machine instruction (typically `lock xadd` on x86) that guarantees atomicity at the level of the memory/cache bus, but only for a simple operation like `+=`. It is fast compared to a generic lock, but it is still a hardware synchronization at every iteration.

**`sum_table_critical`** — a generic critical section (lock/mutex), able to protect any block of code, not just an operation on a scalar. It is the most versatile tool but also the heaviest: each thread has to acquire then release the lock at every iteration, which strongly serializes the execution.

### What you should observe

The expected order, from the fastest to the slowest: `reduction` < `atomic` < `critical`, with `unprotected` giving a very short time but a **wrong result** (to compare with `reel_sum`, computed analytically).

```
The sum unprotected of the table elements is: 781249375000
Time unprotected: 0.00211305 s
The sum with reduction of the table elements is: 49999995000000
Time reduction: 0.00221462 s
The sum with atomic of the table elements is: 49999995000000
Time atomic: 0.699232 s
The sum with critical of the table elements is: 49999995000000
Time critical: 2.66539 s
The expected sum is: 49999995000000
```

`atomic` is about 300 times slower than `reduction` here, and `critical` 4 times slower than `atomic`. The wrong value is also instructive: 781 249 375 000 is *exactly* the sum of 0 + 1 + ... + 1 249 999, i.e. the first eighth of the array. With `-O2`, the compiler keeps `sum` in a register during the loop: each thread reads `sum = 0` at the beginning, accumulates its own block, and writes its partial sum once at the end, overwriting the others. In this run, thread 0 happened to write last.

### Compile and run

```bash
g++ -fopenmp -O2 1_OpenMP_comparaison.cpp -o 1_OpenMP_comparaison
./1_OpenMP_comparaison
```

---

## `2_OpenMP_scheduling.cpp` — scheduling policies

### Goal

Fill an array where `table[i]` is the number of steps of the Collatz sequence starting from `i+1`, in parallel, with four policies to distribute the iterations among the threads.

### Why the Collatz sequence?

The cost of `collatz_steps(i)` varies enormously and unpredictably with `i`: some numbers converge to 1 in a few steps, others need hundreds. It is an **irregular** workload — exactly what is needed to make the effect of scheduling visible. With a loop where every iteration costs the same, all the schedules would give almost identical times and the demo would be pointless.

### The 4 functions

| Function | OpenMP clause | Decided when | Chunk size |
|---|---|---|---|
| `fill_static` | `static` (default) | before the loop | `n / nb_threads`, one contiguous block per thread |
| `fill_static_chunk` | `static, 64` | before the loop | fixed, distributed round-robin |
| `fill_dynamic` | `dynamic, 64` | at run time | fixed, distributed on demand |
| `fill_guided` | `guided` | at run time | decreasing (large at the beginning, small at the end) |

**`fill_static`** — the default schedule. The iteration space is cut into `P` contiguous blocks (`P` = number of threads), and each thread receives exactly one block, decided once and for all before the loop starts. Zero overhead at run time, but if the "heavy" iterations gather in one block (which easily happens with Collatz), that thread becomes the bottleneck while the others wait, idle.

**`fill_static_chunk`** — the same "decided in advance" logic, but with small chunks of fixed size distributed round-robin among the threads (thread 0 gets chunks 0, P, 2P..., thread 1 gets chunks 1, P+1... etc). This mixes heavy and light iterations better between threads, still without any synchronization during the loop.

**`fill_dynamic`** — the chunks are distributed on demand, at run time: as soon as a thread has finished its chunk, it comes back to take the next one from a shared queue. Very good load balancing for an irregular workload like this one, but each distribution costs a (small) synchronization on the queue.

**`fill_guided`** — same principle as `dynamic` (distribution at run time from a shared queue), but the chunk size starts large and decreases geometrically as the loop progresses. Fewer distributions than `dynamic` (hence less overhead), while still adapting finely towards the end of the loop — which is precisely where imbalance costs the most waiting time.

### What you should observe

The 4 functions produce the **same total sum** (checked by `sum_table` at the end of `main`) — the schedule changes *how* the work is shared among threads, never *what* is computed. The times, however, differ: the default `static` is generally the most imbalanced on this kind of workload, `dynamic` and `guided` compensate at the price of a little overhead, and `static, chunk=64` often sits in between.

```
static (default)  : sum = 277182223 | time = 0.426237 s
static, chunk=64  : sum = 277182223 | time = 0.365946 s
dynamic, chunk=64 : sum = 277182223 | time = 0.294723 s
guided             : sum = 277182223 | time = 0.368121 s
All sums equal ? yes
```

### Compile and run

```bash
g++ -fopenmp -O2 2_OpenMP_scheduling.cpp -o 2_OpenMP_scheduling
./2_OpenMP_scheduling
```

---

## `3_OpenMP_numa_affinity.cpp` — thread placement and first touch

### Goal

On a machine with several NUMA nodes (typically a server with 2 sockets), each socket has its own RAM: accessing the memory of the *other* socket goes through an interconnect (Intel UPI, AMD Infinity Fabric) and is slower. Two questions then become important: **where do the threads run** (part 1), and **where do the memory pages live** (part 2).

```
   +--------------- node 0 ---------------+      +--------------- node 1 ---------------+
   |  core 0  core 1  core 2  core 3      |      |  core 4  core 5  core 6  core 7      |
   |              local RAM 0             | <==> |              local RAM 1             |
   +--------------------------------------+ UPI  +--------------------------------------+
            fast access: local              slower access: remote (NUMA factor)
```

Run `lscpu` or `numactl --hardware` to know how many NUMA nodes your machine has and which cores belong to which node.

### Part 1 — thread placement (`OMP_PLACES`, `proc_bind`)

- A **place** is a set of logical CPUs a thread may run on. `OMP_PLACES` defines the list of places: `threads` (one hardware thread each), `cores` (one physical core each), `sockets`, or an explicit list like `"{0,1},{2,3}"`.
- The **binding policy** (`OMP_PROC_BIND` variable, or the `proc_bind` clause of a region) decides how the threads of a team are spread over these places:
  - `close`: as close as possible to the master thread (and to each other) → good when threads exchange a lot of data, they share caches.
  - `spread`: as far apart as possible → each thread gets its own share of caches and memory bandwidth, good when each thread mostly works on its own data.
  - `false`: no binding, the OS can migrate the threads at any time.
- **Trap:** when neither `OMP_PLACES` nor `OMP_PROC_BIND` is set, GCC's runtime starts with `OMP_PROC_BIND = 'FALSE'` (check it with `OMP_DISPLAY_ENV=true`), and the OpenMP specification says that the `proc_bind` clauses are then **ignored**. That is why a plain `./3_OpenMP_numa_affinity` shows several threads reported on the same logical CPU, whatever the clause. Setting `OMP_PLACES` turns binding on:

```bash
OMP_PLACES=threads OMP_NUM_THREADS=4 ./3_OpenMP_numa_affinity
```

On the 4-core laptop, logical CPUs 0 and 4 are the two hyper-threads of the same physical core (see `/sys/devices/system/cpu/cpu0/topology/thread_siblings_list`). With 4 threads:

```
  [close] thread 0 -> core 0       [spread] thread 0 -> core 0
  [close] thread 1 -> core 4       [spread] thread 1 -> core 1
  [close] thread 2 -> core 1       [spread] thread 2 -> core 2
  [close] thread 3 -> core 5       [spread] thread 3 -> core 3
```

`close` packs the 4 threads on 2 physical cores (both hyper-threads of cores 0 and 1), `spread` uses 4 different physical cores. `OMP_DISPLAY_ENV=true` prints the list of places actually used, here `OMP_PLACES = '{0},{4},{1},{5},{2},{6},{3},{7}'`.

### Part 2 — first-touch allocation policy

On Linux, `new double[n]` only reserves *virtual* addresses: no physical page is allocated until it is written for the first time ("touched"), and it is then allocated on the NUMA node of the thread that touched it. So **the thread that initializes an array decides where it physically lives** — which often matters more for performance than the parallelization of the computation itself.

- `init_serial` (bad): the master thread touches everything, so all the pages end up on its node; later, the threads running on the other node only make remote accesses, and the whole computation is limited by the memory bandwidth of a single node.
- `init_parallel` (good): the array is touched in parallel with the **same** `schedule(static)` distribution as the computation, so each thread "owns" (in the NUMA sense) exactly the pages it will work on afterwards.

### What you should observe

On a single-socket laptop, both `process()` times are close (everything is local anyway), as the program itself says. On a multi-socket node, the parallel initialization makes `process()` much faster — a factor close to 2 on a two-socket node is common for this kind of memory-bound loop. Combine it with binding (`OMP_PROC_BIND=spread`, `OMP_PLACES=cores`) so that the threads don't migrate away from the pages they touched.

### Compile and run

```bash
g++ -fopenmp -O2 3_OpenMP_numa_affinity.cpp -o 3_OpenMP_numa_affinity
OMP_PLACES=threads OMP_NUM_THREADS=4 ./3_OpenMP_numa_affinity
```

---

## `4_OpenMP_data_sharing.cpp` — shared and private variables

### Goal

Understand which variables are shared by all the threads and which ones each thread owns, and see a classic real-life data race and how to fix it.

### Key concepts

In a parallel region, each variable is either **shared** (one single copy, seen by every thread) or **private** (one copy per thread). The default rules:

| Variable | Default status |
|---|---|
| declared **outside** the region | shared |
| declared **inside** the region | private |
| index of a loop shared with `omp for` | private |

and the clauses that change them:

| Clause | Each thread gets... | After the region |
|---|---|---|
| `shared(x)` | the single original `x` | — |
| `private(x)` | a new, **uninitialized** copy | the original `x` is unchanged |
| `firstprivate(x)` | a copy **initialized** with the value of `x` before the region | the original `x` is unchanged |
| `lastprivate(x)` | a private copy | `x` receives the value of the sequentially **last** iteration |
| `reduction(op:x)` | a copy initialized with the neutral element of `op` | the copies are combined into `x` (see example 1) |
| `default(none)` | — | no default rule at all: the status of every variable used in the region must be written explicitly |

### Part 1 — `private`, `firstprivate`, `lastprivate`

```
  [private]      thread 3 : its x = 3
  [private]      ...
  [private]      after the region, x = 42 (unchanged)
  [firstprivate] thread 2 : its x = 44
  [firstprivate] thread 0 : its x = 42
  [firstprivate] ...
  [firstprivate] after the region, x = 42 (unchanged)
  [lastprivate]  after the loop, x = 9.94987 = sqrt(99)
```

With `private`, each copy starts uninitialized (reading it before writing it is a bug); with `firstprivate`, every copy starts at 42. In both cases, the original variable is untouched. With `lastprivate`, `x` gets the value of iteration `i = 99`, whatever thread executed it.

### Part 2 — a real bug: the shared scratch buffer

A **median filter** replaces each value by the median of its 5 neighbours — it removes "spikes" from a noisy signal much better than an average. The 5 values are copied into a small scratch buffer, `window`, and sorted. The buggy version is *exactly* the sequential code plus one pragma:

```cpp
double window[5];                 // declared OUTSIDE the parallel region -> shared!
#pragma omp parallel for
for (int i = 2; i < n - 2; ++i) {
    for (int k = 0; k < 5; ++k) window[k] = in[i - 2 + k];
    out[i] = median_of_5(window);
}
```

All the threads copy their values into the **same** 5 cells and sort them at the same time, overwriting each other's data:

```
shared window (BUG)   : 87035 wrong values | time = 0.524197 s
private(window)       : 0 wrong values | time = 0.0555806 s
default(none), local  : 0 wrong values | time = 0.0474533 s
```

- The result is **wrong**, and different at every run. With `OMP_NUM_THREADS=1` the bug disappears: a race needs at least two threads, which is why such bugs often slip through the tests.
- The buggy version is also **10 times slower**: every write of a thread into `window` invalidates the corresponding cache line in the caches of the other cores, so the line keeps bouncing from core to core (cache coherence traffic).
- A data race is *undefined behaviour*: anything can happen. With `std::sort` instead of the small insertion sort used here, the program does not just give wrong values, it **crashes**: `std::sort` relies on invariants of the data that another thread breaks while it runs, and ends up writing outside the array.

**Two fixes**: `private(window)` (a private array is a whole array per thread), or — more readable — declaring `window` inside the loop body, in the smallest possible scope. Adding `default(none)` turns any forgotten variable into a **compile-time error** instead of a silent race; removing `in` from the `shared(...)` list gives:

```
error: 'in' not specified in enclosing 'parallel'
```

### Compile and run

```bash
g++ -fopenmp -O2 4_OpenMP_data_sharing.cpp -o 4_OpenMP_data_sharing
./4_OpenMP_data_sharing
```

---

## `5_OpenMP_worksharing_sync.cpp` — work-sharing and synchronization constructs

### Goal

`parallel for` is not the only way to share work among the threads of a team, and implicit barriers are not the only way to make them wait for each other. This file goes through the other **work-sharing** constructs (`single`, `sections`) and **synchronization** constructs (`barrier`, `nowait`, `master`, `ordered`, locks).

### Part 1 — `single`, `barrier`, `nowait`: a parallel prefix sum

The inclusive prefix sum (or *scan*) computes `out[i] = in[0] + ... + in[i]`. Each iteration needs the previous one (`out[i] = out[i-1] + in[i]`), so a plain `parallel for` can't do it. The classic parallel algorithm works in 3 phases separated by synchronization points:

```
in :  | block of thread 0 | block of thread 1 | block of thread 2 |
1. in parallel : each thread computes the total of its block            T0      T1      T2
2. one thread  : offset of block t = total of blocks 0..t-1             0       T0      T0+T1
3. in parallel : each thread scans its block, starting from its offset
```

Each construct has a precise role in it:

- `single` allocates the array of offsets: one thread does it, and the **implicit barrier** at the end of `single` guarantees nobody uses the array before it exists. Phase 2 is also a `single`.
- `nowait` removes the implicit barrier at the end of the phase-1 loop: to go on, a thread only needs *its own* block total...
- ...but phase 2 reads the totals of **all** the blocks, hence the explicit `#pragma omp barrier`. Without it, the thread doing phase 2 could read a total that has not been written yet.
- Phase 3 relies on a guarantee of the specification: two `schedule(static)` loops with the same number of iterations, in the same parallel region, give each thread **the same block**.

Don't expect a big speedup here: a prefix sum does almost no computation per element, so its speed is limited by the memory bandwidth (which a single core already uses a good part of), and the parallel algorithm reads the input twice. The point of this part is the synchronization. (OpenMP 5.0 also provides a ready-made `scan` directive.)

### Part 2 — `master` vs `single`

| | `master` (renamed `masked` in OpenMP 5.1) | `single` |
|---|---|---|
| executed by | always thread 0 | the first thread that gets there |
| barrier at the end | no, the others don't wait | yes (unless `nowait`) |

```
  region 0 : master -> thread 0, single -> thread 0
  ...
  region 3 : master -> thread 0, single -> thread 2
```

### Part 3 — `sections`: different jobs at the same time

`parallel for` splits **one** job (a loop) among the threads: *data decomposition*. `sections` gives a **different** job to each thread: *functional (task) decomposition*. Here, three statistics of the same array (mean, min/max, count above 0.5) are computed at the same time by three threads. The parallelism is limited by the number of sections (3, whatever the number of threads), and the slowest section sets the pace: tasks (example 6) generalize this idea.

### Part 4 — `ordered`

The expensive part of each iteration runs in parallel in any order, but the `ordered` block is executed in the order of the iterations — typically to print or write results in the right order:

```
  i = 0 -> 15.0859   (computed by thread 4)
  i = 1 -> 15.779   (computed by thread 3)
  i = 2 -> 16.1845   (computed by thread 6)
  ...
```

### Part 5 — `critical` vs locks vs privatization

Each bin of a histogram stores 3 fields (count, sum, max). Updating a bin is a **compound** operation, which `atomic` can't protect (it only handles one simple operation on one scalar). Three solutions:

| Version | Synchronization |
|---|---|
| `critical` | one global lock: all updates are serialized, even those of different bins |
| one `omp_lock_t` per bin | two threads only wait for each other when they update the same bin |
| private copies + merge | each thread fills its own histogram with no synchronization at all, then merges it once |

The lock API: `omp_init_lock` → `omp_set_lock` (wait, then take it) → `omp_unset_lock` (release it) → `omp_destroy_lock` (`omp_test_lock` tries without waiting).

```
critical (1 global lock) : 2.45321 s | correct ? yes
one lock per bin         : 1.72557 s | correct ? yes
private copies + merge   : 0.0310152 s | correct ? yes
```

The per-bin locks are only slightly faster than `critical`: taking and releasing a lock are atomic operations on a memory word, and the 64 locks fit in a few cache lines that keep bouncing between the cores. Privatization is **80 times** faster than `critical`: 8 merges instead of 10 million locked updates. This is exactly what `reduction` does under the hood, and the general lesson of this file: *the best synchronization is the one you don't need*.

The values are multiples of 1/1024, so that every sum is exact in double precision whatever the order of the additions: the histograms can be compared with `==`.

### Compile and run

```bash
g++ -fopenmp -O2 5_OpenMP_worksharing_sync.cpp -o 5_OpenMP_worksharing_sync
./5_OpenMP_worksharing_sync
```

---

## `6_OpenMP_tasks.cpp` — task parallelism

### Goal

`parallel for` needs a loop whose number of iterations is known when it starts. Recursive algorithms (divide and conquer, tree traversals), linked lists, `while` loops or graphs of jobs don't look like that: OpenMP 3.0 introduced **tasks** for them.

### Key concepts

A task is a piece of work (code + its data) that a thread packages and puts in a pool; any thread of the team can pick it up and execute it, now or later. The usual pattern:

```cpp
#pragma omp parallel        // creates the team of threads
#pragma omp single          // ONE thread creates the tasks...
{
    #pragma omp task        // ...and the others execute them
    work();
}                           // implicit barrier: all the tasks are finished here
```

- `taskwait` waits for the **child** tasks created so far by the current task (`taskgroup` waits for all the descendants).
- **Data-sharing trap**: in a task, the local variables of the enclosing function are `firstprivate` by default. A task that must return a result through a variable needs it `shared`, otherwise it writes into its own private copy, which disappears with the task.

### Part 1 — recursive tasks and granularity (Fibonacci)

`fib(n) = fib(n-1) + fib(n-2)` is a terribly slow way to compute Fibonacci numbers, but it is the simplest recursive problem: each call creates two tasks and waits for them.

```
fib(25) sequential           = 75025 | time = 0.000664102 s
fib(25) one task per call    = 75025 | time = 0.443148 s  <- much SLOWER
fib(38) sequential           = 39088169 | time = 0.349826 s
fib(38) tasks with a cutoff  = 39088169 | time = 0.188448 s
```

Creating, queuing and scheduling a task costs **one or two microseconds** (0.44 s for the ~240 000 tasks of `fib(25)`), while `fib(2)` costs a nanosecond: with one task per call, the program is hundreds of times slower than the sequential one! The fix is a **cutoff**: below a threshold, the function calls the sequential version instead of creating tasks. Choosing the granularity of the tasks is *the* key to their performance (the `if()` and `final()` clauses do a similar job, less efficiently).

### Part 2 — traversing a linked list

A linked list can only be walked node after node (`p = p->next`), so `parallel for` can't share it. One thread walks the list (cheap) and creates one task per node; the processing of the nodes (expensive) happens in parallel. `firstprivate(p)` makes each task keep the address of *its* node, even after the loop has moved `p` forward.

```
sequential : sum of results = 3289.78 | time = 0.413673 s
tasks      : sum of results = 3289.78 | time = 0.135593 s
```

### Part 3 — task dependencies

`depend(in: x)` means "this task reads `x`" and `depend(out: x)` "this task writes `x`". The runtime builds the graph of the tasks from these clauses and starts each task as soon as its inputs are ready. Each task "works" for 0.2 s:

```
                  A (writes a)
        .---------'----------.
        v                    v
   B (reads a,          C (reads a,        <- B and C run at the same time
      writes b)            writes c)
        '---------.----------'
                  v
        D (reads b and c, writes d)
```

```
  A done at t = 0.202749 s
  C done at t = 0.402903 s
  B done at t = 0.405657 s
  D done at t = 0.605756 s
  d = 12 (expected 12), total = 0.611649 s instead of 0.8 s sequentially
```

B and C finish at the same time (in any order), D waits for both: the total is 3 steps instead of 4.

### Compile and run

```bash
g++ -fopenmp -O2 6_OpenMP_tasks.cpp -o 6_OpenMP_tasks
./6_OpenMP_tasks
```

---

## `7_OpenMP_simd_collapse.cpp` — vectorization and nested loops

### Goal

Use the **SIMD** units of the cores (a level of parallelism *inside* each core, different from threads), combine them with threads, and parallelize nested loops with `collapse`.

### Part 1 — `simd`: vectorizing a reduction

Every modern core can apply the same operation to several numbers with **one** instruction (SIMD, in Flynn's taxonomy): a 256-bit AVX2 register holds 4 doubles, a 512-bit AVX-512 register holds 8.

```
scalar :  s += x[0]*y[0]   then   s += x[1]*y[1]   then ...          (1 product per instruction)
simd   :  [s0 s1 s2 s3] += [x0 x1 x2 x3] * [y0 y1 y2 y3]             (4 products per instruction)
          and s = s0 + s1 + s2 + s3 at the end
```

Compilers vectorize loops by themselves (GCC from `-O3`, or `-O2` since GCC 12), but only when they can prove it doesn't change the result. A floating-point sum can **never** be vectorized automatically (without `-ffast-math`): `(a + b) + c` is not exactly `a + (b + c)` in floating point, and the vectorized version adds the numbers in another order. `#pragma omp simd reduction(+:s)` tells the compiler: vectorize, I accept the reordering.

```
scalar : 2.46 s | dot = 0.92351626475972015
simd   : 0.647 s | dot = 0.92351626475971949
speedup of simd : 3.81 | difference of the results : -6.66e-16 (other order of the additions)
```

The speedup is close to 4 (4 doubles per AVX2 register), and the two results differ in the last digits: both are "correct", they are rounded differently. To check what the compiler vectorized: `-fopt-info-vec-optimized` (GCC), `-Rpass=loop-vectorize` (Clang), `-qopt-report` (Intel), `-Minfo=vect` (NVIDIA).

### Part 2 — threads **and** SIMD: `parallel for simd`

The logistic map `v -> r v (1 - v)`, iterated 200 times for a million values of `r` (this is how its famous *bifurcation diagram* is drawn), performs 600 floating-point operations per number read from memory: the best case for both threads and SIMD. `parallel for simd` first splits the iterations among the threads (MIMD), then each thread vectorizes its own chunk (SIMD). `declare simd` asks the compiler to also generate a vector version of a function, used when a simd loop calls it without inlining it.

```
1 thread , scalar : 1.14 s
1 thread , simd   : 0.302 s | speedup 3.78
8 threads, scalar : 0.269 s | speedup 4.24
8 threads, simd   : 0.0886 s | speedup 12.9
```

The two levels of parallelism multiply: ~4 from SIMD × ~3-4 from threads. The results are bit-for-bit identical, because each value is computed with exactly the same operations (no reduction here).

### Part 3 — `collapse`: sharing nested loops

An RGB image stored channel by channel is processed with two nested loops: 3 channels × 4 million pixels. `parallel for` only shares the **outer** loop: 3 iterations for 8 threads, so 5 threads wait. `collapse(2)` merges the two loops into a single loop of 12 million iterations, shared among all the threads. The loops must be perfectly nested (nothing between the two `for`) and their bounds independent of each other.

```
parallel for (outer loop only) : 0.24 s | threads with work : 3 / 8
parallel for collapse(2)       : 0.154 s | threads with work : 8 / 8
```

On this 4-core laptop the gain is modest (3 busy cores vs 4 cores + hyper-threads), but on a 64-core node it would be 3 threads against 64.

### Compile and run

`-march=native` lets the compiler use the widest vector instructions of your processor (AVX2, AVX-512...) instead of the SSE2 baseline (2 doubles per register): without it, the SIMD speedups are halved.

```bash
g++ -fopenmp -O2 -march=native 7_OpenMP_simd_collapse.cpp -o 7_OpenMP_simd_collapse
./7_OpenMP_simd_collapse
```

---

## `8_OpenMP_gpu_offload.cpp` — offloading to a GPU

### Goal

Since version 4.0, OpenMP can also **offload** computations to an accelerator, most of the time a GPU, with the same directive-based approach. The same ideas are found in OpenACC (`../OpenACC`) and, written by hand, in CUDA (`../CUDA`).

### Key concepts

The GPU has its **own memory**: data must be copied to it before a computation and back afterwards, through the PCIe bus. These copies are often much slower than the computation itself, so the main skill of GPU programming is to **move as little data as possible**.

```
   +---------+   PCIe (~16-64 GB/s)   +--------------------+
   |   CPU   | <====================> |        GPU         |
   |   RAM   |                        | VRAM (~0.5-3 TB/s) |
   +---------+                        +--------------------+
      host                                   device
```

**The directives and the GPU hierarchy:**

| Directive | Role | CUDA equivalent |
|---|---|---|
| `target` | move the execution of the region to the device | kernel launch |
| `teams` | create a *league* of teams that can't synchronize with each other | grid of blocks |
| `distribute` | share the iterations among the teams | `blockIdx` |
| `parallel for` | share each team's iterations among its threads | `threadIdx` |
| `simd` | vector lanes (on some GPUs) | — |

**The `map` clauses** describe the data movements of a `target` region:

| Clause | Before the region | After the region |
|---|---|---|
| `map(to: x[0:n])` | copy host → device | — |
| `map(from: x[0:n])` | allocate only | copy device → host |
| `map(tofrom: x[0:n])` | copy host → device | copy device → host |
| `map(alloc: x[0:n])` | allocate only | — |

**Part 1** is a SAXPY (`y = a x + y`, the "hello world" of accelerators). **Part 2** is a dot product with a reduction, and shows a trap: in a target region, a scalar is `firstprivate` by default (copied to the device, never copied back). OpenMP 5.0 maps a reduction variable `tofrom` automatically, but older compilers (GCC 11 for example) follow OpenMP 4.5 and silently return 0: writing `map(tofrom: sum)` is portable and explicit.

**Part 3** solves a 1D heat equation for 200 time steps, two ways:
- **naive**: each kernel has its own `map` clauses, so the arrays travel to the GPU and back at every step (24 MB per kernel, ~5 GB in total);
- **`target data`**: the arrays are copied once when entering the data region and once when leaving it; the kernels inside find them already on the device. `target update from(...)` brings the data back to the host on demand — here to monitor the temperature during the run. `declare target` compiles a function for the device so that kernels can call it.

### What you should observe

On a machine without a GPU (or with a compiler without offloading support), the target regions run on the host and the program says so. The map clauses then copy nothing (host and "device" are the same memory), so both versions of part 3 take similar times: the gap only shows up on a real GPU, where the naive version spends most of its time in PCIe transfers.

```
number of offload devices : 0
target regions run on     : the HOST (CPU fallback)
...
naive (map at every step) :
    step 50 : u[10] = 4.60441
    step 100 : u[10] = 15.8165
    step 150 : u[10] = 24.8958
    step 200 : u[10] = 31.7915
...
max difference between the two versions : 0
```

The monitored temperatures match the analytical solution of the heat equation, `100 * erfc(x / (2 sqrt(alpha t)))`: 4.6, 15.7, 25.0 and 31.7.

### Compile and run

```bash
nvc++ -mp=gpu -O2 8_OpenMP_gpu_offload.cpp -o 8_OpenMP_gpu_offload                               # NVIDIA HPC SDK
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -O2 8_OpenMP_gpu_offload.cpp -o 8_OpenMP_gpu_offload  # Clang
g++ -fopenmp -foffload=nvptx-none -O2 8_OpenMP_gpu_offload.cpp -o 8_OpenMP_gpu_offload             # GCC + gcc-offload-nvptx
g++ -fopenmp -O2 8_OpenMP_gpu_offload.cpp -o 8_OpenMP_gpu_offload                                  # CPU fallback
./8_OpenMP_gpu_offload
```

With GCC's CPU fallback, each target region pays the creation of a new team of threads, made worse by the threads of the previous team that keep spinning for a while: `OMP_WAIT_POLICY=passive ./8_OpenMP_gpu_offload` makes this example about 4 times faster on the CPU.

---

## Cheat sheet

| Construct | Purpose |
|---|---|
| `#pragma omp parallel` | create a team of threads |
| `#pragma omp for` / `parallel for` | share the iterations of a loop |
| `schedule(static\|dynamic\|guided [,chunk])` | how the iterations are distributed |
| `collapse(n)` | share `n` perfectly nested loops as one |
| `private`, `firstprivate`, `lastprivate`, `shared`, `default(none)` | data-sharing attributes |
| `reduction(op:var)` | private copies combined at the end |
| `atomic` | one simple update of one scalar, done atomically |
| `critical` | a block executed by one thread at a time |
| `barrier` | wait for all the threads of the team |
| `nowait` | remove the implicit barrier of a work-sharing construct |
| `single` / `master` | a block executed by one thread (with / without barrier) |
| `sections` / `section` | different blocks of code for different threads |
| `ordered` | a part of a loop executed in sequential order |
| `task`, `taskwait`, `taskgroup`, `depend` | task parallelism |
| `simd`, `declare simd` | vectorization |
| `target`, `teams`, `distribute`, `map`, `target data` | offloading to an accelerator |

| Function | Returns |
|---|---|
| `omp_get_thread_num()` | id of the calling thread in its team |
| `omp_get_num_threads()` | size of the current team |
| `omp_get_max_threads()` | size of the next team |
| `omp_get_num_procs()` | number of logical cores available |
| `omp_get_wtime()` | wall-clock time in seconds |
| `omp_set_num_threads(n)` | sets the size of the next teams |
| `omp_init_lock`, `omp_set_lock`, `omp_unset_lock`, `omp_destroy_lock` | explicit locks |
| `omp_get_num_devices()`, `omp_is_initial_device()` | offloading devices |

---

## Going further

- Fix the number of threads for reproducible comparisons: `OMP_NUM_THREADS=4 ./2_OpenMP_scheduling`.
- In the scheduling file, vary `chunk_size` (try 1, 8, 64, 512...) and look at the size from which the overhead of `dynamic` exceeds the gain in load balancing.
- Vary `n` in the files: the larger `n`, the less the synchronization overhead (per iteration) weighs compared to the computation itself.
- Combine the files: nothing prevents adding `schedule(dynamic)` to a loop that also uses `reduction` — the two clauses answer independent questions (how to share the work / how to aggregate a shared result).
- In `5_OpenMP_worksharing_sync.cpp`, give each lock its own cache line (`struct alignas(64) PaddedLock { omp_lock_t lock; };`) and measure the effect of *false sharing* on the per-bin locks.
- In `6_OpenMP_tasks.cpp`, vary the cutoff of `fib_tasks_cutoff`: too low, the task overhead dominates; too high, there are not enough tasks to keep every thread busy.
- In `7_OpenMP_simd_collapse.cpp`, compile without `-march=native`, then with `-O3`, and compare the vectorization reports (`-fopt-info-vec-optimized`).
- The same "1D/2D heat equation" problem is solved with MPI (`../MPI`) and OpenACC (`../OpenACC`): comparing the codes shows what each model makes explicit.

## References

- OpenMP specifications and reference cards: <https://www.openmp.org/specifications/>
- LLNL OpenMP tutorial: <https://hpc-tutorials.llnl.gov/openmp/>
- B. Chapman, G. Jost, R. van der Pas, *Using OpenMP*, MIT Press, 2007 — and R. van der Pas, E. Stotzer, C. Terboven, *Using OpenMP — The Next Step*, MIT Press, 2017 (tasks, SIMD, offloading).
