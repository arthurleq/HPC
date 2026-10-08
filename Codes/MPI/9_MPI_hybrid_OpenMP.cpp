#include <iostream>
#include <iomanip>
#include <vector>

#include <mpi.h>
#include <omp.h>

// sched_getcpu() : on which logical CPU is the calling thread running ?
// (Linux / glibc only, see also ../OpenMP/3_OpenMP_numa_affinity.cpp)
#ifdef __linux__
#include <sched.h>
#endif 

// mpic++ -fopenmp -O2 9_MPI_hybrid_OpenMP.cpp -o 9_MPI_hybrid_OpenMP
// OMP_NUM_THREADS=2 mpirun -np 2 --map-by slot:PE=2 ./9_MPI_hybrid_OpenMP
// (see the README for the placement options of mpirun)

// HYBRID programming : MPI between the nodes (distributed memory), OpenMP
// inside each node (shared memory). Two levels of parallelism, like the
// machines themselves, which are clusters of multi-core nodes :
//
//         node 0                                node 1
//   +---------------------------+         +---------------------------+
//   |  MPI process (rank 0)     |         |  MPI process (rank 1)     |
//   |  OpenMP threads 0 1 2 3   | <=====> |  OpenMP threads 0 1 2 3   |
//   |  one shared memory        | network |  one shared memory        |
//   +---------------------------+         +---------------------------+
//
// Compared to one MPI process per core : fewer processes, so fewer and
// bigger messages, and less memory duplicated in each process (ghost
// cells, tables, buffers...).
//
// MPI must be told that threads exist : MPI_Init_thread replaces
// MPI_Init, with the level of thread support the program needs :
//   MPI_THREAD_SINGLE      a single thread exists
//   MPI_THREAD_FUNNELED    several threads, but only the master thread
//                          calls MPI (the most common case, used here)
//   MPI_THREAD_SERIALIZED  any thread may call MPI, but one at a time
//   MPI_THREAD_MULTIPLE    any thread may call MPI at any time
// The library returns the level it really provides, which may be lower.


// ---------------------------------------------------------------
// PART 1 : who runs where ?
// ---------------------------------------------------------------
// Each thread of each process reports the logical CPU it runs on. It shows
// whether the processes and their threads are placed as intended : this
// is the first thing to check on a new machine (see the README).
void part1_placement(int rank, int size) {

    int nthreads = omp_get_max_threads();
    int max_threads;
    MPI_Allreduce(&nthreads, &max_threads, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    // cpu[t] = logical CPU of thread t of this process (-1 : no such thread)
    std::vector<int> cpu(max_threads, -1);
    #pragma omp parallel
    {
#ifdef __linux__
        cpu[omp_get_thread_num()] = sched_getcpu();
#else
        cpu[omp_get_thread_num()] = -2;  // unknown on this platform
#endif
    }

    // only the master thread calls MPI (MPI_THREAD_FUNNELED)
    std::vector<int> all(rank == 0 ? size * max_threads : 0);
    MPI_Gather(cpu.data(), max_threads, MPI_INT, all.data(), max_threads, MPI_INT, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            std::cout << "rank " << r << " :";
            for (int t = 0; t < max_threads; ++t) {
                if (all[r * max_threads + t] != -1) {
                    std::cout << "  thread " << t << " -> cpu " << all[r * max_threads + t];
                }
            }
            std::cout << std::endl;
        }
    }
}


// ---------------------------------------------------------------
// PART 2 : two levels of parallelism -> pi again
// ---------------------------------------------------------------
// pi = integral from 0 to 1 of 4 / (1 + x^2) dx (see example 4)
//   level 1 (MPI)    : each process takes a contiguous block of rectangles
//   level 2 (OpenMP) : the threads of the process share its block
// then an OpenMP reduction inside each process, and an MPI reduction
// between the processes.
double compute_pi(int rank, int size, long n) {

    long first = rank * (n / size);
    long last = (rank == size - 1) ? n : first + n / size;
    const double h = 1.0 / n;

    double local_sum = 0.0;
    #pragma omp parallel for reduction(+:local_sum)
    for (long i = first; i < last; ++i) {
        double x = (i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }

    double local_pi = local_sum * h, pi = 0.0;
    MPI_Reduce(&local_pi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return pi;
}


int main(int argc, char** argv) {

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::cerr << "this MPI library does not support threads" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        const char* levels[] = {"SINGLE", "FUNNELED", "SERIALIZED", "MULTIPLE"};
        std::cout << size << " MPI processes x " << omp_get_max_threads()
                  << " OpenMP threads, thread support provided : MPI_THREAD_"
                  << levels[provided] << std::endl;
        std::cout << std::endl << "=== Part 1 : placement ===" << std::endl;
    }
    part1_placement(rank, size);

    if (rank == 0) std::cout << std::endl << "=== Part 2 : pi with MPI + OpenMP ===" << std::endl;

    const long n = 2000000000L;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    double pi = compute_pi(rank, size, n);
    double t = MPI_Wtime() - t0;

    if (rank == 0) {
        std::cout << std::setprecision(15) << "pi ~ " << pi << std::setprecision(3)
                  << " | time = " << t << " s" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
