# MPI — Message Passing Interface

MPI is the programming model associated with **distributed memory**: a standard that lets several **processes** cooperate by exchanging **messages**. Each process has its own private memory, possibly on a different machine (a *node* of a cluster): no variable is shared, and every piece of data that another process needs must be sent to it explicitly.

Its main advantage is **scalability**: to get more computing power you add nodes, and each new node brings more cores *and* more memory. That is why all the large simulation codes running on supercomputers use MPI, often combined with OpenMP or a GPU model inside each node. MPI is a library (functions `MPI_*`), not a language extension; the main implementations are Open MPI, MPICH and their derivatives (Intel MPI, MVAPICH, Cray MPICH). Although it was designed for distributed memory, MPI works perfectly well inside a single machine: the processes of a node then exchange their messages through shared memory, which is how the examples of this folder run on a laptop.

All the programs follow the **SPMD** model (Single Program, Multiple Data): `mpirun -np 4 ./program` starts 4 copies of the same program, and each copy uses its number, its **rank**, to decide what to do and on which part of the data.

| File | Topic | Main functions |
|---|---|---|
| [`1_MPI_hello_world.cpp`](1_MPI_hello_world.cpp) | processes, ranks, the SPMD model | `MPI_Init`, `MPI_Comm_rank`, `MPI_Comm_size`, `MPI_Finalize` |
| [`2_MPI_point_to_point.cpp`](2_MPI_point_to_point.cpp) | messages between two processes, latency and bandwidth, deadlocks | `MPI_Send`, `MPI_Recv`, `MPI_Probe`, `MPI_Sendrecv` |
| [`3_MPI_non_blocking.cpp`](3_MPI_non_blocking.cpp) | non-blocking communications, overlapping | `MPI_Isend`, `MPI_Irecv`, `MPI_Wait*`, `MPI_Issend` |
| [`4_MPI_collectives.cpp`](4_MPI_collectives.cpp) | operations involving all the processes | `MPI_Bcast`, `MPI_Scatter(v)`, `MPI_Gather(v)`, `MPI_(All)reduce`, `MPI_Alltoall` |
| [`5_MPI_derived_datatypes.cpp`](5_MPI_derived_datatypes.cpp) | sending non-contiguous data and structures | `MPI_Type_vector`, `MPI_Type_create_subarray`, `MPI_Type_create_struct` |
| [`6_MPI_communicators_topologies.cpp`](6_MPI_communicators_topologies.cpp) | groups of processes, process grids | `MPI_Comm_split`, `MPI_Cart_create`, `MPI_Cart_shift` |
| [`7_MPI_heat_2D.cpp`](7_MPI_heat_2D.cpp) | **case study**: domain decomposition and halo exchange | everything above |
| [`8_MPI_one_sided.cpp`](8_MPI_one_sided.cpp) | remote memory access, dynamic load balancing | `MPI_Win_allocate`, `MPI_Put`, `MPI_Get`, `MPI_Fetch_and_op` |
| [`9_MPI_hybrid_OpenMP.cpp`](9_MPI_hybrid_OpenMP.cpp) | MPI between nodes + OpenMP inside a node | `MPI_Init_thread`, process and thread placement |

---

## Compiling and running

```bash
make                         # builds every example of the folder
make 4_MPI_collectives       # builds only one of them
make clean                   # removes the executables
mpirun -np 4 ./4_MPI_collectives
```

or by hand:

```bash
mpicxx -O2 4_MPI_collectives.cpp -o 4_MPI_collectives
mpirun -np 4 ./4_MPI_collectives
```

- `mpicxx` (or `mpic++`) is a **wrapper**: it calls the usual C++ compiler with the include and library paths of MPI. `mpicxx --showme` (Open MPI) or `mpicxx -show` (MPICH) prints the real command.
- `mpirun -np N` (or `mpiexec`) starts N processes. By default Open MPI refuses to start more processes than physical cores: add `--oversubscribe` (or `--use-hwthread-cpus` to count hyper-threads) to run more of them on a laptop.
- On a cluster, the processes are usually started by the job scheduler, e.g. `srun -n 64 ./program` with SLURM, and spread over several nodes.
- The output of the processes is forwarded to your terminal by `mpirun`: the lines of different ranks can appear in any order. That is why the examples let rank 0 print most of the results, after collecting them.

The sample outputs below come from a laptop (Intel Core i7-1185G7, 4 cores / 8 hardware threads) with Open MPI 4.1.2: all the processes run on the same machine and communicate through shared memory, not through a network.

---

## `1_MPI_hello_world.cpp` — processes and ranks

### Goal

Start several processes, see that they are really separate programs, and learn the 4 functions present in every MPI program.

### Key concepts

```
   mpirun -np 4 ./1_MPI_hello_world
        |
        +--> process of rank 0  (its own memory)
        +--> process of rank 1  (its own memory)     they communicate
        +--> process of rank 2  (its own memory)     by messages only
        +--> process of rank 3  (its own memory)
```

- `MPI_Init` must be called before any other MPI function, `MPI_Finalize` at the very end.
- A **communicator** is a group of processes that can talk to each other. `MPI_COMM_WORLD` contains all the processes started by `mpirun`.
- `MPI_Comm_rank` gives the rank of the process in the communicator (0 to N-1), `MPI_Comm_size` the number of processes.
- Each process has its own copy of every variable: `value = 10 * rank` is a different variable in each process. There is no `shared` here, unlike OpenMP.

### What you should observe

```
hello from rank 2 / 4 (pid 83577) on node my-laptop
hello from rank 1 / 4 (pid 83576) on node my-laptop
rank 1 : my value is 10
hello from rank 3 / 4 (pid 83578) on node my-laptop
hello from rank 0 / 4 (pid 83575) on node my-laptop
rank 0 : MPI standard 3.1, implementation : Open MPI v4.1.2
...
```

Four different process ids, the same node name (they all run on the laptop), and lines in a different order at each run.

### Compile and run

```bash
mpicxx -O2 1_MPI_hello_world.cpp -o 1_MPI_hello_world
mpirun -np 4 ./1_MPI_hello_world
mpirun -np 8 --oversubscribe ./1_MPI_hello_world
```

---

## `2_MPI_point_to_point.cpp` — sending and receiving messages

### Goal

Send a message from one process to another, measure what it costs, and discover the most classic bug of message passing: the deadlock.

### Key concepts

A message is made of a **buffer** (address, number of elements, MPI datatype) and an **envelope** (source, destination, tag, communicator):

```cpp
MPI_Send(buffer, count, MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
MPI_Recv(buffer, count, MPI_DOUBLE, source,      tag, MPI_COMM_WORLD, &status);
```

A receive matches a send when the source, the tag and the communicator match; `MPI_ANY_SOURCE` and `MPI_ANY_TAG` are wildcards. The `status` tells who sent the message and with which tag. When the size of a message is unknown, `MPI_Probe` describes the next incoming message without receiving it, and `MPI_Get_count` gives its number of elements.

Both calls are **blocking**: `MPI_Recv` returns when the message is in the buffer, `MPI_Send` when the buffer can be reused — the data has been sent, *or copied into an internal buffer of MPI*. This last point is the key to part 3.

### Part 2 — latency and bandwidth (ping-pong)

The time to send a message of `n` bytes is well described by `T(n) = latency + n / bandwidth`. Rank 0 sends a message to rank 1, which sends it back, many times:

```
   message size |  time / message |   bandwidth
           8 B |         0.49 us |    0.02 GB/s
          64 B |         0.46 us |    0.14 GB/s
         512 B |         0.94 us |    0.54 GB/s
        4096 B |         4.18 us |    0.98 GB/s
       32768 B |        11.52 us |    2.85 GB/s
      262144 B |        51.84 us |    5.06 GB/s
     2097152 B |       322.64 us |    6.50 GB/s
    16777216 B |      3719.59 us |    4.51 GB/s
```

- Small messages cost the **latency** (~0.5 µs here, 1-2 µs between two nodes on an InfiniBand network) whatever their size: sending 8 bytes or 64 bytes takes the same time.
- Large messages reach the **bandwidth** of the link (a few GB/s here, through the memory of the laptop; 10-50 GB/s on a modern HPC network).
- Consequence: it is much better to send **one** big message than many small ones (aggregate your data).

### Part 3 — the ring and the deadlock

Each rank sends a big message (8 MB) to its right neighbour and receives one from its left neighbour:

```
      rank 0 ---> rank 1 ---> rank 2 ---> rank 3
        ^                                   |
        +-----------------------------------+
```

The natural code — every rank calls `MPI_Send`, then `MPI_Recv` — **deadlocks**: a message this big can't be stored in an internal buffer, so `MPI_Send` waits for the receiver to call `MPI_Recv`... but every rank is stuck in its own `MPI_Send`. Run `mpirun -np 4 ./2_MPI_point_to_point deadlock` to see the program hang (Ctrl+C to stop it).

The vicious part: with **small** messages, MPI copies the data into an internal buffer and `MPI_Send` returns at once (the *eager* protocol, used below a threshold of a few kilobytes, which depends on the implementation and the network). The same code then works perfectly on small test cases... and hangs on the real, large ones (the *rendezvous* protocol). The program shows two correct solutions:

1. **break the symmetry**: even ranks send then receive, odd ranks receive then send;
2. **`MPI_Sendrecv`**: sends and receives at the same time and lets MPI handle the ordering (non-blocking communications, in example 3, are the other classic solution).

### Compile and run

```bash
mpicxx -O2 2_MPI_point_to_point.cpp -o 2_MPI_point_to_point
mpirun -np 4 ./2_MPI_point_to_point
mpirun -np 4 ./2_MPI_point_to_point deadlock     # hangs on purpose: Ctrl+C
```

---

## `3_MPI_non_blocking.cpp` — non-blocking communications

### Goal

Start communications without waiting for them, to avoid deadlocks and to keep working while messages are in flight.

### Key concepts

`MPI_Isend` and `MPI_Irecv` **start** a communication and return immediately with an `MPI_Request`. Until the request is completed, the buffer must not be touched (no writing into a buffer being sent, no reading of a buffer being received). A request is completed with:

| Function | Behaviour |
|---|---|
| `MPI_Wait(&req, &status)` | blocks until the communication is complete |
| `MPI_Test(&req, &flag, &status)` | checks without blocking (`flag = 1` if complete) |
| `MPI_Waitall(n, reqs, statuses)` | waits for all of them |
| `MPI_Waitany(n, reqs, &index, &status)` | waits for the first one to complete |

### What you should observe

```
=== Part 1 : non-blocking ring exchange ===
Isend / Irecv / Waitall : correct, no deadlock

=== Part 2 : overlapping ===
blocking     (MPI_Ssend)  : rank 0 done after 2.00 s
non-blocking (MPI_Issend) : rank 0 done after 1.00 s

=== Part 3 : MPI_Waitany ===
  after 0.20 s : result of rank 3 = 300.00
  after 0.40 s : result of rank 2 = 200.00
  after 0.60 s : result of rank 1 = 100.00
```

- **Part 1**: the ring of example 2 with `MPI_Irecv` + `MPI_Isend` + `MPI_Waitall` never deadlocks, whatever the order of the calls and the size of the messages. Posting the receive first is a good habit: the message can then go straight into its final buffer.
- **Part 2**: rank 1 is busy for 1 s before it can receive; rank 0 has to send it a message and has 1 s of independent work. With a blocking synchronous send (`MPI_Ssend` completes only when the receive has started), rank 0 waits for 1 s then works for 1 s; with `MPI_Issend`, it works while the message waits, and the final `MPI_Wait` returns almost immediately.
- **Part 3**: a manager/workers pattern. Rank 0 posts one `MPI_Irecv` per worker and `MPI_Waitany` hands back the results **in order of arrival** (the last ranks finish first here), not in the order of the ranks.

**A word of caution about overlapping.** Here, what is hidden is a *waiting time*. Hiding the *transfer* of a large message behind a computation also requires the data to move while the process computes: on a cluster this is done by the network card (RDMA), but many MPI libraries only make progress inside MPI calls, so long computations are sometimes sprinkled with `MPI_Test` calls. Inside a single machine, the "transfer" is a memory copy done by a CPU core anyway.

### Compile and run

```bash
mpicxx -O2 3_MPI_non_blocking.cpp -o 3_MPI_non_blocking
mpirun -np 4 ./3_MPI_non_blocking
```

---

## `4_MPI_collectives.cpp` — collective operations

### Goal

Use the operations that involve **all** the processes of a communicator: broadcasting, distributing, collecting and reducing data.

### Key concepts

Every process of the communicator must call the same collective, in the same order (if one process skips it, the others wait forever). Collectives are implemented with optimized algorithms (trees, rings...): a broadcast to P processes takes about log₂(P) steps instead of P-1 sends from the root. **Always prefer a collective to a hand-written loop of sends and receives.**

Part 1 prints, for 4 processes, what each collective does with the data — its output *is* the diagram:

```
MPI_Bcast     : the root's buffer {1, 2, 3, 4} is copied everywhere
    rank 0 :    1    2    3    4
    rank 1 :    1    2    3    4
    ...
MPI_Scatter   : the root's {0, 10, 20, ...} is dealt out
    rank 0 :    0
    rank 1 :   10
    rank 2 :   20
    rank 3 :   30
MPI_Gather    : the root collects rank*rank from everybody
    rank 0 :    0    1    4    9    (the other ranks receive nothing)
MPI_Allgather : everybody collects rank*rank from everybody
    rank 0 :    0    1    4    9
    ...
MPI_Reduce    : sum of (rank + 1) over all the ranks = 10 (result on the root only)
MPI_Allreduce : max of (rank + 1), known by everybody
    rank 0 :    4
    ...
MPI_Alltoall  : before, rank i has 10*i + j at position j
    rank 0 :    0    1    2    3
    rank 1 :   10   11   12   13
    rank 2 :   20   21   22   23
    rank 3 :   30   31   32   33
                after, rank j has 10*i + j at position i
    rank 0 :    0   10   20   30
    rank 1 :    1   11   21   31
    rank 2 :    2   12   22   32
    rank 3 :    3   13   23   33
```

| Collective | Data movement |
|---|---|
| `MPI_Bcast` | one → all (same data) |
| `MPI_Scatter` | one → all (a different piece for each) |
| `MPI_Gather` | all → one |
| `MPI_Allgather` | all → all (everybody gets everything) |
| `MPI_Reduce` | all → one, combined with an operation (`MPI_SUM`, `MPI_MAX`, `MPI_MIN`, `MPI_PROD`, `MPI_LAND`...) |
| `MPI_Allreduce` | all → all, combined |
| `MPI_Alltoall` | all → all, a different piece for each (a transposition) |
| `MPI_Barrier` | no data: nobody leaves before everybody has arrived |

### Part 2 — computing π (`Bcast` + `Reduce`)

π is the integral of `4 / (1 + x²)` between 0 and 1. Rank 0 decides the number of rectangles and broadcasts it, each rank sums its share of the rectangles, and `MPI_Reduce` adds the partial results on rank 0:

```
pi ~ 3.14159265358993 | error = 1.34e-13 | time = 0.255 s with 4 processes
```

(1.03 s with 1 process: a speedup of 4, since the processes only communicate twice.)

### Part 3 — distributed statistics (`Scatterv`, `Allreduce`, `Gatherv`, `MAXLOC`)

Rank 0 owns a data set of N = 1 000 003 values. It distributes it, the mean and the standard deviation are computed together, every process normalizes its part (`z = (x - mean) / std`), and the result is collected back:

- N is **not** a multiple of the number of processes: the "v" versions (`MPI_Scatterv`, `MPI_Gatherv`) take one count per process and the position (*displacement*) of each part in the root's buffer.
- `MPI_Allreduce` gives the mean to **every** process, which needs it for the next step (the standard deviation, then the normalization).
- `MPI_MAXLOC` reduces (value, index) pairs: it finds the maximum **and** where it is.

```
counts per rank : 250001 250001 250001 250000   (N = 1000003)
mean     : parallel 49.9952017 | sequential 49.9952017
std dev  : parallel 10.00158893 | sequential 10.00158893
maximum  : 99.33819013 at index 514292 (data[514292] = 99.33819013)
normalized data : max difference with the sequential version = 1.51e-13
```

The tiny difference with the sequential version comes from the order of the additions, which differs when the sum is split among processes (floating-point addition is not associative): the result of a parallel reduction can change slightly with the number of processes.

### Compile and run

```bash
mpicxx -O2 4_MPI_collectives.cpp -o 4_MPI_collectives
mpirun -np 4 ./4_MPI_collectives
```

---

## `5_MPI_derived_datatypes.cpp` — derived datatypes

### Goal

Send non-contiguous data (a column, a block of a matrix) or a C++ structure in **one** message, without copying it by hand into a temporary buffer.

### Key concepts

A derived datatype describes a memory layout. It is built with a constructor, committed with `MPI_Type_commit` before use, and freed with `MPI_Type_free`.

**A column — `MPI_Type_vector(count, blocklength, stride, oldtype, &newtype)`.** In C/C++, matrices are stored row by row: a row is contiguous, but a column is made of `rows` elements separated by `cols` elements:

```
    [  0   1   2   3   4   5 ]
    [  6   7   8   9  10  11 ]     column 2 = elements 2, 8, 14, 20 :
    [ 12  13  14  15  16  17 ]     4 blocks of 1 element, stride 6
    [ 18  19  20  21  22  23 ]
```

The receive type does not have to be the send type: only the sequence of basic types must match. The column can be received into a contiguous array of 4 doubles, or directly into another column of a matrix.

**A block — `MPI_Type_create_subarray`.** Describes a sub-block of a multi-dimensional array by its shape and its starting corner: exactly what a process sends to its neighbours in a domain decomposition (example 7).

**A structure — `MPI_Type_create_struct`.** A structure mixes several types, and the compiler inserts **padding** to align its fields:

```
   | id | pad |  position[0]  position[1]  position[2]  |  mass  |
   0    4     8                                         32       40 bytes
```

`offsetof()` gives the real position of each field, and `MPI_Type_create_resized` sets the *extent* of the type to `sizeof(Particle)` so that MPI finds each element of an array of particles at the right place:

```
particle type : size = 36 bytes of data, extent = 40 bytes (= sizeof(Particle), padding included)
    particle 7 at (0, 1, 2), mass 1.5
    particle 8 at (3, 4, 5), mass 2.5
    particle 9 at (6, 7, 8), mass 3.5
```

### Compile and run

```bash
mpicxx -O2 5_MPI_derived_datatypes.cpp -o 5_MPI_derived_datatypes
mpirun -np 2 ./5_MPI_derived_datatypes
```

---

## `6_MPI_communicators_topologies.cpp` — communicators and process grids

### Goal

Create groups of processes to run collectives on a subset of them, and arrange the processes in a grid where everyone knows their neighbours.

### Key concepts

A communicator is a group of processes plus a private *context*: the messages of one communicator never mix with those of another (this is why libraries work on their own copy, made with `MPI_Comm_dup`).

**`MPI_Comm_split(comm, color, key, &newcomm)`**: the processes that give the same `color` end up in the same new communicator, ranked by `key`. With `color = row`, we get one communicator per row of a grid of processes, and an `MPI_Allreduce` on it only involves the processes of that row:

```
grid of 3 rows x 2 columns
   world     row     col  rank in  rank in  sum of   sum of
    rank                  row comm col comm  my row  my column
       0       0       0       0       0       1       6
       1       0       1       1       0       1       9
       2       1       0       0       1       5       6
       ...
```

**Cartesian topologies**: `MPI_Dims_create` chooses a balanced grid shape, `MPI_Cart_create` builds a communicator with this shape (each dimension periodic or not), `MPI_Cart_coords` gives the coordinates of a rank and `MPI_Cart_shift` the ranks of its neighbours. At a non-periodic border, the missing neighbour is **`MPI_PROC_NULL`**: sending to it or receiving from it does nothing and returns at once, so the code needs **no special case** for the processes at the border:

```
    rank     row     col   north   south    west    east
       0       0       0       -       2       1       1
       1       0       1       -       3       0       0
       2       1       0       0       4       3       3
       ...
       4       2       0       2       -       5       5
       5       2       1       3       -       4       4
```

(north/south is not periodic, west/east is: with 2 columns, the west and east neighbours are the same process.)

### Compile and run

```bash
mpicxx -O2 6_MPI_communicators_topologies.cpp -o 6_MPI_communicators_topologies
mpirun -np 6 --oversubscribe ./6_MPI_communicators_topologies
```

---

## `7_MPI_heat_2D.cpp` — case study: domain decomposition

### Goal

Put everything together in the way almost every simulation code (CFD, climate, structural mechanics...) runs on a cluster: **domain decomposition** with **halo exchanges**. This is also what CFD software like OpenFOAM does when a case is split with `decomposePar` and run with `mpirun`.

### The problem

Heat diffusion in a square plate of 512 × 512 cells: a hot disk (100 °C) in the middle of a cold plate whose borders are kept at 0 °C. At each time step, every cell moves towards the average of its 4 neighbours (explicit finite differences for the heat equation):

```
T_new[i][j] = T[i][j] + alpha * (T[i-1][j] + T[i+1][j] + T[i][j-1] + T[i][j+1] - 4 T[i][j])
```

### The parallel algorithm

1. **Decomposition.** The processes form a 2D Cartesian grid (example 6) and the plate is cut into blocks, one per process. Each process only stores and updates its block (even when 512 is not a multiple of the number of processes).
2. **Ghost cells.** To update the cells at the edge of its block, a process needs cells owned by its neighbours. Each block is therefore surrounded by a layer of *ghost cells*, copies of the neighbours' edge cells:

   ```
           +---------------------+
           | g g g g g g g g g g |   g = ghost cells : received from the
           | g o o o o o o o o g |       4 neighbours before each step
           | g o o o o o o o o g |
           | g o o o o o o o o g |   o = cells owned and updated by
           | g g g g g g g g g g |       this process
           +---------------------+
   ```

3. **Halo exchange.** Before each step, 4 `MPI_Sendrecv` (north, south, west, east) refresh the ghost cells. Rows are contiguous; columns use an `MPI_Type_vector` (example 5). At the border of the plate, the neighbour is `MPI_PROC_NULL`: the ghost cells are never overwritten and keep their value 0, they *are* the boundary condition.
4. **Computation.** Each process runs exactly the sequential update on its block — literally the same function as the sequential reference.
5. **Global diagnostics.** Every 1000 steps, `MPI_Reduce` computes the total heat (`MPI_SUM`) and the maximum temperature (`MPI_MAX`). A reduction is a global synchronization: it is not done at every step.
6. **Gathering.** At the end, each process sends the interior of its block (a subarray type that skips the ghost cells) and rank 0 receives each block directly at its place in the global array (another subarray type, one per block).

### What you should observe

```
plate of 512 x 512 cells, 5000 time steps, 4 processes as a 2 x 2 grid (blocks of about 256 x 256 cells)
  step  1000 : total heat = 2287200, max temperature = 99.99
  step  2000 : total heat = 2287200, max temperature = 98.94
  step  3000 : total heat = 2287200, max temperature = 95.18
  step  4000 : total heat = 2287196, max temperature = 89.72
  step  5000 : total heat = 2287162, max temperature = 83.80
parallel   : 1.011 s
sequential : 3.391 s   -> speedup 3.356
max difference with the sequential result : 0.000

temperature map (' ' = 0 ... '@' = 100) :
  |                         ..............                         |
  |                     .....:::::::::::.....                      |
  |                   ....::::----------::::...                    |
  |                  ...:::--=====++====---:::...                  |
  |                 ...::--==++++****++++==--::...                 |
  |                ...::--=+++***####***++==--::...                |
  |                ..::--=++**##########**+==--::..                |
  |               ...::-==++**##%%%%%%##**++=--::...               |
  |               ...::-==++**##%%%%%%##**++=--::...               |
  |                ..::--==+**#########***+==--::..                |
  |                ...::--==++***####***++==--::...                |
  |                 ...::--==++++****+++===--::...                 |
  |                  ...:::---==========---:::...                  |
  |                    ...::::----------::::...                    |
  |                      .....::::::::::.....                      |
  |                         ..............                         |
```

- The total heat is **conserved** as long as the heat has not reached the cold borders, then starts to leak out: a good physical sanity check.
- The parallel result is **bit-for-bit identical** to the sequential one, with any number of processes: each cell is computed with the same operations on the same values, only the place where it is computed changes.
- Speedup on the 4-core laptop:

| processes | grid | time | speedup |
|---|---|---|---|
| 1 | 1 × 1 | 3.47 s | 0.97 |
| 2 | 2 × 1 | 1.76 s | 1.95 |
| 3 | 3 × 1 | 1.27 s | 2.69 |
| 4 | 2 × 2 | 1.01 s | 3.36 |

The amount of computation of a block grows like its **area**, its communications like its **perimeter**: the bigger the blocks, the more efficient the parallelization (*surface-to-volume ratio*). With a much smaller plate, the communications would dominate. The same problem is solved on a GPU in [`../OpenACC/6_OpenACC_heat_2D.cpp`](../OpenACC/6_OpenACC_heat_2D.cpp).

### Compile and run

```bash
mpicxx -O2 7_MPI_heat_2D.cpp -o 7_MPI_heat_2D
mpirun -np 4 ./7_MPI_heat_2D
```

---

## `8_MPI_one_sided.cpp` — one-sided communications (RMA)

### Goal

Read and write the memory of another process **without its participation**, and use atomic operations to balance the work dynamically.

### Key concepts

A process exposes a part of its memory in a **window** (`MPI_Win_allocate` allocates the memory and creates the window); the other processes can then access it directly:

```
     two-sided (Send / Recv)                one-sided (Put / Get)
   origin            target              origin            target
   Send  --------->  Recv                Put   --------->  [window]    (the target does nothing)
```

| Function | Effect |
|---|---|
| `MPI_Put` | write into the window of a target |
| `MPI_Get` | read from the window of a target |
| `MPI_Accumulate` | combine data into the window (`MPI_SUM`, `MPI_MAX`...), atomically |
| `MPI_Fetch_and_op` | atomically read a value and update it (e.g. `old = counter; counter += 1`) |

The price to pay: synchronization becomes explicit. RMA operations happen inside **epochs**, and their result is only guaranteed once the epoch is closed:
- **active target**: `MPI_Win_fence` (collective, all the processes of the window call it);
- **passive target**: `MPI_Win_lock` / `MPI_Win_unlock` (only the origin is involved, the target keeps working).

### What you should observe

- **Part 1**: every rank `MPI_Put`s its result into rank 0's window — the same result as an `MPI_Gather`, but rank 0 never receives anything explicitly.
- **Part 2**: every rank `MPI_Get`s a value from the window of its right neighbour.
- **Part 3**: 40 tasks of increasing cost (2 to 80 ms) must be shared. A static split gives the most expensive tasks to the last rank; with a **shared task counter** in the window of rank 0, each process atomically takes the next task (`MPI_Fetch_and_op`) as soon as it is free — the MPI equivalent of OpenMP's `schedule(dynamic)`, with no process dedicated to distributing the work:

```
static split (blocks of consecutive tasks) : finished after 0.710 s
dynamic (shared task counter)            : finished after 0.440 s (ideal : total work / 4 = 0.410 s)
  rank 0 did 10 tasks, busy 0.400 s
  rank 1 did 10 tasks, busy 0.420 s
  rank 2 did 10 tasks, busy 0.380 s
  rank 3 did 10 tasks, busy 0.440 s
```

### Compile and run

```bash
mpicxx -O2 8_MPI_one_sided.cpp -o 8_MPI_one_sided
mpirun -np 4 ./8_MPI_one_sided
```

---

## `9_MPI_hybrid_OpenMP.cpp` — hybrid MPI + OpenMP

### Goal

Combine the two levels of parallelism of a cluster: MPI between the nodes, OpenMP threads inside each node — and learn to check that processes and threads run where they should.

### Key concepts

```
         node 0                                node 1
   +---------------------------+         +---------------------------+
   |  MPI process (rank 0)     |         |  MPI process (rank 1)     |
   |  OpenMP threads 0 1 2 3   | <=====> |  OpenMP threads 0 1 2 3   |
   |  one shared memory        | network |  one shared memory        |
   +---------------------------+         +---------------------------+
```

Compared to one MPI process per core, the hybrid model uses fewer processes: fewer and bigger messages, and less memory duplicated in each process (ghost cells, tables, buffers).

MPI must be told that threads exist: `MPI_Init_thread` replaces `MPI_Init` and asks for a level of thread support (the library answers with the level it really provides):

| Level | Meaning |
|---|---|
| `MPI_THREAD_SINGLE` | only one thread exists |
| `MPI_THREAD_FUNNELED` | several threads, but only the master thread calls MPI (the most common case, used here) |
| `MPI_THREAD_SERIALIZED` | any thread may call MPI, but never two at the same time |
| `MPI_THREAD_MULTIPLE` | any thread may call MPI at any time |

### What you should observe: the placement trap

`mpirun` **binds** each process to a set of cores, and the OpenMP threads of a process can only run on the cores of their process. With 2 processes, Open MPI binds each process to **one core** by default, so its 2 threads share that single core:

```bash
OMP_NUM_THREADS=2 mpirun -np 2 ./9_MPI_hybrid_OpenMP
```
```
rank 0 :  thread 0 -> cpu 0  thread 1 -> cpu 4
rank 1 :  thread 0 -> cpu 5  thread 1 -> cpu 1
pi ~ 3.1415926535898 | time = 2.54 s
```

On this laptop, logical CPUs 0 and 4 are the two hyper-threads of the **same** physical core (as are 1 and 5): each process really uses one core. Giving 2 cores (*processing elements*) to each process fixes it:

```bash
OMP_NUM_THREADS=2 mpirun -np 2 --map-by slot:PE=2 ./9_MPI_hybrid_OpenMP
```
```
rank 0 :  thread 0 -> cpu 0  thread 1 -> cpu 5
rank 1 :  thread 0 -> cpu 6  thread 1 -> cpu 3
pi ~ 3.1415926535898 | time = 1.23 s
```

Twice as fast, with the same code. **Always check the placement** on a new machine (`--report-bindings` with Open MPI). With SLURM, the usual recipe is:

```bash
#SBATCH --ntasks-per-node=4        # MPI processes per node
#SBATCH --cpus-per-task=8          # cores per process
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun ./9_MPI_hybrid_OpenMP
```

(When running on several nodes with `mpirun`, export the variable to the remote processes with `-x OMP_NUM_THREADS`.)

### Compile and run

```bash
mpicxx -fopenmp -O2 9_MPI_hybrid_OpenMP.cpp -o 9_MPI_hybrid_OpenMP
OMP_NUM_THREADS=2 mpirun -np 2 --map-by slot:PE=2 ./9_MPI_hybrid_OpenMP
```

---

## Cheat sheet

| Function | Purpose |
|---|---|
| `MPI_Init`, `MPI_Init_thread`, `MPI_Finalize` | start / stop MPI |
| `MPI_Comm_rank`, `MPI_Comm_size` | who am I, how many are we |
| `MPI_Wtime` | wall-clock time in seconds |
| `MPI_Send`, `MPI_Recv`, `MPI_Sendrecv`, `MPI_Ssend` | blocking point-to-point |
| `MPI_Isend`, `MPI_Irecv`, `MPI_Wait`, `MPI_Waitall`, `MPI_Waitany`, `MPI_Test` | non-blocking point-to-point |
| `MPI_Probe`, `MPI_Get_count` | inspect an incoming message |
| `MPI_Bcast`, `MPI_Scatter(v)`, `MPI_Gather(v)`, `MPI_Allgather` | collective data movements |
| `MPI_Reduce`, `MPI_Allreduce` | collective reductions |
| `MPI_Alltoall`, `MPI_Barrier` | all-to-all exchange, synchronization |
| `MPI_Type_vector`, `MPI_Type_create_subarray`, `MPI_Type_create_struct`, `MPI_Type_commit`, `MPI_Type_free` | derived datatypes |
| `MPI_Comm_split`, `MPI_Comm_dup`, `MPI_Comm_free` | communicators |
| `MPI_Dims_create`, `MPI_Cart_create`, `MPI_Cart_coords`, `MPI_Cart_shift` | Cartesian topologies |
| `MPI_Win_allocate`, `MPI_Win_fence`, `MPI_Win_lock`, `MPI_Put`, `MPI_Get`, `MPI_Fetch_and_op` | one-sided communications |
| `MPI_Abort` | stop all the processes after a fatal error |

---

## Going further

- In `7_MPI_heat_2D.cpp`, replace the 4 `MPI_Sendrecv` by `MPI_Irecv`/`MPI_Isend`, update the **interior** of the block (which needs no ghost cell) while the messages are in flight, then the edges after `MPI_Waitall`: this is the classic way to overlap communications and computations.
- Still in example 7, measure the time spent in communications (`MPI_Wtime` around the halo exchange) as a function of the plate size and the number of processes, and relate it to the surface-to-volume ratio.
- Non-blocking collectives (`MPI_Iallreduce`...), neighbourhood collectives on topologies (`MPI_Neighbor_alltoall`), persistent requests (`MPI_Send_init`), shared-memory windows (`MPI_Win_allocate_shared` with `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`), parallel I/O (`MPI_File_*`).
- Profiling tools show the time spent in each MPI call and the messages exchanged: mpiP, Score-P + Vampir, Intel Trace Analyzer, Arm/Linaro Forge.

## References

- MPI Forum, the official standard: <https://www.mpi-forum.org/docs/>
- LLNL MPI tutorial: <https://hpc-tutorials.llnl.gov/mpi/>
- W. Gropp, E. Lusk, A. Skjellum, *Using MPI*, 3rd edition, MIT Press, 2014 — and W. Gropp, T. Hoefler, R. Thakur, E. Lusk, *Using Advanced MPI*, MIT Press, 2014 (one-sided communications, hybrid programming).
