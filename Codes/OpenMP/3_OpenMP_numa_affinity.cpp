#include <iostream>

// initialisation of OpenMP environment variables
#include <omp.h>

// sched_getcpu() is a Linux/glibc extension : it lets a thread ask
// "which physical CPU core am I currently running on ?". It's the
// easiest way to actually SEE where OpenMP places each thread.
// Not available on macOS / Windows -> falls back to -1 there.
#ifdef __linux__
#include <sched.h>
#endif


// ---------------------------------------------------------------
// PART 1 : thread placement (OMP_PROC_BIND / proc_bind clause)
// ---------------------------------------------------------------
// Run "numactl --hardware" or "lscpu" beforehand to know how many
// NUMA nodes your machine has and which cores belong to which node.
// On a single-socket laptop everything below will still run, but
// there is only one node to place threads on, so don't expect any
// visible difference between the 3 placements.

// print, for the calling thread, its OpenMP id and its current
// physical core
void report_placement(const char* label) {
    int tid = omp_get_thread_num();
#ifdef __linux__
    int cpu = sched_getcpu();
#else
    int cpu = -1; // sched_getcpu unavailable on this platform
#endif
    #pragma omp critical
    {
        std::cout << "  [" << label << "] thread " << tid
                  << " -> core " << cpu << std::endl;
    }
}

// no proc_bind clause : threads follow OMP_PROC_BIND (env variable),
// or are left free to migrate between cores if it isn't set. This is
// the "worst case" for NUMA : the OS scheduler can move a thread
// away from the memory it touched first, turning local accesses into
// remote ones over time.
void placement_default() {
    #pragma omp parallel
    {
        report_placement("default");
    }
}

// proc_bind(close) : threads are packed as close as possible to the
// master thread (and to each other). Good when threads collaborate
// tightly and exchange data often -> minimizes inter-thread latency.
void placement_close() {
    #pragma omp parallel proc_bind(close)
    {
        report_placement("close");
    }
}

// proc_bind(spread) : threads are spread out as evenly as possible
// across the available places (cores / NUMA nodes). Good when each
// thread mostly works on its own data -> maximizes aggregate memory
// bandwidth instead of concentrating everyone on one node.
void placement_spread() {
    #pragma omp parallel proc_bind(spread)
    {
        report_placement("spread");
    }
}


// ---------------------------------------------------------------
// PART 2 : first-touch allocation policy
// ---------------------------------------------------------------
// On Linux, "new double[n]" only reserves virtual address space :
// no physical memory page is actually allocated until it is written
// to for the first time ("touched"). And it gets allocated on the
// NUMA node local to whichever thread does that first touch. So WHO
// initializes the array decides WHERE it physically lives -- often
// more important for performance than the parallelization of the
// computation itself.

// BAD first-touch : a single thread (the master, no pragma) touches
// every element. Every physical page ends up local to that one
// thread's NUMA node, no matter how the later computation is split.
void init_serial(long n, double* table) {
    for (long i = 0; i < n; ++i) {
        table[i] = 0.0;
    }
}

// GOOD first-touch : touch the array in parallel, using the SAME
// static distribution that will be used later to process it. Each
// thread ends up "owning", in the NUMA sense, exactly the pages it
// will work on again afterwards.
void init_parallel(long n, double* table) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
        table[i] = 0.0;
    }
}

// the actual work : touch every element again (read + write), with
// the exact same static distribution as init_parallel, so each
// thread re-accesses the pages it may (or may not) have initialized
// itself.
void process(long n, double* table) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
        table[i] = table[i] * 2.0 + 1.0;
    }
}


int main() {

    std::cout << "=== Part 1 : thread placement ===" << std::endl;
    std::cout << "-- default --" << std::endl;
    placement_default();
    std::cout << "-- proc_bind(close) --" << std::endl;
    placement_close();
    std::cout << "-- proc_bind(spread) --" << std::endl;
    placement_spread();

    std::cout << std::endl << "=== Part 2 : first-touch effect ===" << std::endl;

    // ~800 MB with 8-byte doubles : large enough to be split across
    // NUMA nodes if your machine has more than one. Adjust to your
    // available RAM if needed.
    const long n = static_cast<long>(100e6);

    // ---- scenario A : bad first-touch ----
    double* table_a = new double[n];
    init_serial(n, table_a);
    double t0 = omp_get_wtime();
    process(n, table_a);
    double t1 = omp_get_wtime();
    std::cout << "process() after SERIAL init   : " << (t1 - t0) << " s" << std::endl;
    delete[] table_a;

    // ---- scenario B : good first-touch ----
    double* table_b = new double[n];
    init_parallel(n, table_b);
    t0 = omp_get_wtime();
    process(n, table_b);
    t1 = omp_get_wtime();
    std::cout << "process() after PARALLEL init : " << (t1 - t0) << " s" << std::endl;
    delete[] table_b;

    std::cout << std::endl
              << "Note: the gap between the two process() times above only shows up"
              << std::endl
              << "on multi-socket / multi-NUMA-node machines. On a single-socket"
              << std::endl
              << "laptop, expect both times to be close (everything is 'local' anyway)."
              << std::endl;

    return 0;
}
