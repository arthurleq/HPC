#include <iostream>

// initialisation of OpenMP environment variables
#include <omp.h>

// Compute the number of steps of the Collatz sequence starting from n
// until it reaches 1. The cost of this function varies wildly and
// unpredictably with the starting value, which makes it a classic
// example of an IRREGULAR workload -> perfect to illustrate the
// difference between scheduling policies (a well-balanced parallel
// loop is one where every thread finishes at roughly the same time).
int collatz_steps(long start) {
    long n = start;
    int steps = 0;
    while (n != 1) {
        if (n % 2 == 0) {
            n = n / 2;
        } else {
            n = 3 * n + 1;
        }
        ++steps;
    }
    return steps;
}

// fill the table with collatz_steps(i+1) for i in [0, n)
// DEFAULT schedule = static, no chunk size specified
// -> the iteration space is cut into P contiguous blocks (P = number
// of threads). Each thread gets exactly ONE block, decided ONCE,
// before the loop even starts. Zero runtime overhead, but if the
// "heavy" iterations happen to cluster in one block, that thread
// becomes the bottleneck while the others sit idle.
void fill_static(int n, int* table) {

    #pragma omp parallel for schedule(static)

    for (int i = 0; i < n; ++i) {
        table[i] = collatz_steps(i + 1);
    }
}

// static schedule with an EXPLICIT chunk size
// -> the iteration space is cut into small chunks, still handed out
// once, before the loop starts, but distributed round-robin among
// threads (thread 0 gets chunk 0, P, 2P..., thread 1 gets chunk 1,
// P+1... etc). Smaller chunks interleave "heavy" and "light"
// iterations more evenly between threads, still with zero runtime
// synchronization cost.
void fill_static_chunk(int n, int* table, int chunk_size) {

    #pragma omp parallel for schedule(static, chunk_size)

    for (int i = 0; i < n; ++i) {
        table[i] = collatz_steps(i + 1);
    }
}

// dynamic schedule with an explicit chunk size
// -> chunks are handed out to threads ON DEMAND, at runtime: as soon
// as a thread finishes its chunk, it comes back and grabs the next
// available one from a shared queue. Excellent load balancing for
// irregular workloads like this one, but every hand-out requires a
// (lightweight) synchronization on the shared queue.
void fill_dynamic(int n, int* table, int chunk_size) {

    #pragma omp parallel for schedule(dynamic, chunk_size)

    for (int i = 0; i < n; ++i) {
        table[i] = collatz_steps(i + 1);
    }
}

// guided schedule
// -> same idea as dynamic (runtime hand-out from a shared queue), but
// the chunk size starts large and shrinks geometrically as the loop
// progresses. This means fewer hand-outs than dynamic (less overhead)
// while still adapting finely near the end of the loop, which is
// usually where imbalance costs the most idle time.
void fill_guided(int n, int* table) {

    #pragma omp parallel for schedule(guided)

    for (int i = 0; i < n; ++i) {
        table[i] = collatz_steps(i + 1);
    }
}

// simple sequential helper, only used to check that every schedule
// produces the exact same result: scheduling changes HOW the work is
// split among threads, never WHAT is computed.
long sum_table(int n, int* table) {
    long sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += table[i];
    }
    return sum;
}


int main() {

    // size of the table (kept smaller than the memory-management
    // example, since collatz_steps is much more expensive per
    // iteration than a simple addition)
    const int n = static_cast<int>(2e6);

    int* table = new int[n];

    // ---- static, default (contiguous blocks, one per thread) ----
    double t0 = omp_get_wtime();
    fill_static(n, table);
    double t1 = omp_get_wtime();
    long sum_static = sum_table(n, table);
    std::cout << "static (default)  : sum = " << sum_static
              << " | time = " << (t1 - t0) << " s" << std::endl;

    // ---- static, small chunk (round-robin distribution) ----
    t0 = omp_get_wtime();
    fill_static_chunk(n, table, 64);
    t1 = omp_get_wtime();
    long sum_static_chunk = sum_table(n, table);
    std::cout << "static, chunk=64  : sum = " << sum_static_chunk
              << " | time = " << (t1 - t0) << " s" << std::endl;

    // ---- dynamic, small chunk (runtime hand-out) ----
    t0 = omp_get_wtime();
    fill_dynamic(n, table, 64);
    t1 = omp_get_wtime();
    long sum_dynamic = sum_table(n, table);
    std::cout << "dynamic, chunk=64 : sum = " << sum_dynamic
              << " | time = " << (t1 - t0) << " s" << std::endl;

    // ---- guided (runtime hand-out, shrinking chunks) ----
    t0 = omp_get_wtime();
    fill_guided(n, table);
    t1 = omp_get_wtime();
    long sum_guided = sum_table(n, table);
    std::cout << "guided             : sum = " << sum_guided
              << " | time = " << (t1 - t0) << " s" << std::endl;

    // sanity check: every schedule must produce the same total
    bool all_equal = (sum_static == sum_static_chunk) &&
                      (sum_static_chunk == sum_dynamic) &&
                      (sum_dynamic == sum_guided);
    std::cout << "All sums equal ? " << (all_equal ? "yes" : "no") << std::endl;

    // free the allocated memory (no leak)
    delete[] table;
    return 0;
}
