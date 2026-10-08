#include <iostream>
#include <random>

// initialisation of OpenMP environment variables
#include <omp.h>

// g++ -fopenmp -O2 5_OpenMP_worksharing_sync.cpp -o 5_OpenMP_worksharing_sync
// ./5_OpenMP_worksharing_sync

// "parallel for" is not the only way to share work between the threads
// of a team, and the implicit barriers are not the only way to make them
// wait for each other. This file goes through the other WORK-SHARING
// constructs (single, sections) and SYNCHRONIZATION constructs (barrier,
// nowait, master, ordered, locks).

 
// ---------------------------------------------------------------
// PART 1 : single, barrier and nowait -> a parallel prefix sum
// ---------------------------------------------------------------
// Inclusive prefix sum (or "scan") : out[i] = in[0] + in[1] + ... + in[i].
// Sequentially it is a trivial loop, but iteration i needs the result
// of iteration i-1 (out[i] = out[i-1] + in[i]), so a plain "parallel
// for" can't do it. The classic parallel algorithm has 3 phases,
// separated by synchronization points :
//
//   in :  |  block of thread 0  |  block of thread 1  |  block of thread 2  |
//   1. each thread computes the TOTAL of its block (in parallel)
//   2. ONE thread turns the block totals into offsets :
//      offset of block t = total of blocks 0 .. t-1
//   3. each thread scans its block, starting from the offset of the
//      block instead of 0 (in parallel)
// (OpenMP 5.0 also offers a ready-made "scan" directive, but writing it
// by hand is the best way to understand single, barrier and nowait)

// sequential reference
void prefix_sum_seq(int n, const long* in, long* out) {
    long s = 0;
    for (int i = 0; i < n; ++i) {
        s += in[i];
        out[i] = s;
    }
}

void prefix_sum_parallel(int n, const long* in, long* out) {

    // declared outside the region -> shared by the whole team
    long* offset = nullptr;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();

        // single : the block is executed by ONE thread (the first to
        // arrive), the others skip it and WAIT at its end (implicit
        // barrier) -> nobody can use offset before it is allocated
        #pragma omp single
        {
            offset = new long[nth + 1];
            offset[0] = 0;
        }

        // phase 1 : schedule(static) gives each thread ONE contiguous
        // block, in the order of the thread ids. nowait removes the
        // implicit barrier at the end of the loop : to go on, a thread
        // only needs ITS OWN block total, not the others'.
        long block_total = 0;
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i) {
            block_total += in[i];
        }
        offset[tid + 1] = block_total;

        // ... but phase 2 reads the totals of ALL the blocks -> explicit
        // barrier. Without it, the thread doing phase 2 could read a total
        // that another thread has not written yet.
        #pragma omp barrier

        // phase 2 : sequential, but tiny (nth iterations) -> single
        #pragma omp single
        for (int t = 1; t <= nth; ++t) {
            offset[t] += offset[t - 1];
        }
        // implicit barrier of single : the offsets are ready for everybody

        // phase 3 : the SAME static schedule over the SAME number of
        // iterations is guaranteed to give each thread the SAME block as
        // in phase 1, so thread tid starts from the offset of its block
        long running = offset[tid];
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            running += in[i];
            out[i] = running;
        }
    }

    delete[] offset;
}


// ---------------------------------------------------------------
// PART 2 : master vs single
// ---------------------------------------------------------------
// Both execute a block with only one thread, but :
//   master : ALWAYS thread 0, and NO barrier at the end : the other
//            threads don't wait for it (renamed "masked" in OpenMP 5.1)
//   single : the FIRST thread to get there (any id), and an implicit
//            barrier at the end (unless "nowait" is added)
void demo_master_single() {
    for (int run = 0; run < 4; ++run) {

        int master_tid = -1;
        int single_tid = -1;

        #pragma omp parallel num_threads(4)
        {
            #pragma omp master
            master_tid = omp_get_thread_num();

            #pragma omp single
            single_tid = omp_get_thread_num();
        }

        std::cout << "  region " << run << " : master -> thread " << master_tid
                  << ", single -> thread " << single_tid << std::endl;
    }
}


// ---------------------------------------------------------------
// PART 3 : sections -> different jobs at the same time
// ---------------------------------------------------------------
// "parallel for" splits ONE job (a loop) among the threads : data
// decomposition. "sections" gives a DIFFERENT job to each thread :
// functional (or task) decomposition. Each section is executed once, by
// one thread. The parallelism is limited by the number of sections (3
// here, whatever the number of threads), and the slowest section sets
// the pace -> example 6 (tasks) generalizes this idea.
void demo_sections(int n, const double* x) {

    double sum = 0.0;
    double min = x[0], max = x[0];
    long above_half = 0;
    int tid_sum = -1, tid_minmax = -1, tid_count = -1;

    // each section writes its OWN shared variables -> no race
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int i = 0; i < n; ++i) sum += x[i];
            tid_sum = omp_get_thread_num();
        }

        #pragma omp section
        {
            for (int i = 0; i < n; ++i) {
                if (x[i] < min) min = x[i];
                if (x[i] > max) max = x[i];
            }
            tid_minmax = omp_get_thread_num();
        }

        #pragma omp section
        {
            for (int i = 0; i < n; ++i) {
                if (x[i] > 0.5) ++above_half;
            }
            tid_count = omp_get_thread_num();
        }
    }

    std::cout << "  mean       = " << sum / n << "   (section run by thread "
              << tid_sum << ")" << std::endl;
    std::cout << "  min, max   = " << min << ", " << max
              << "   (section run by thread " << tid_minmax << ")" << std::endl;
    std::cout << "  x > 0.5    : " << above_half << " values   (section run by thread "
              << tid_count << ")" << std::endl;
}


// ---------------------------------------------------------------
// PART 4 : ordered -> a sequential part inside a parallel loop
// ---------------------------------------------------------------
// The expensive part of each iteration runs in parallel, in any order,
// but the block marked "ordered" is executed in the order of the
// iterations (0, 1, 2...), like in a sequential loop. Typical use :
// writing results to a file or the screen in the right order.

// a function whose cost depends on i (the last iterations are slower)
double slow_function(int i) {
    double r = 0.0;
    for (long k = 0; k < (i + 1) * 2000000L; ++k) {
        r += 1.0 / (k + 1);
    }
    return r;
}

void demo_ordered() {
    #pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < 8; ++i) {

        double r = slow_function(i);   // in parallel, in any order

        #pragma omp ordered
        std::cout << "  i = " << i << " -> " << r << "   (computed by thread "
                  << omp_get_thread_num() << ")" << std::endl;
    }
}


// ---------------------------------------------------------------
// PART 5 : locks -> protecting a compound update
// ---------------------------------------------------------------
// Histogram where each bin keeps 3 fields. Updating a bin is a COMPOUND
// operation (3 fields at once), which "atomic" can't protect : atomic
// only works on ONE scalar with ONE simple operation.
struct Bin {
    long count = 0;
    double sum = 0.0;
    double max = 0.0;  // the values are in [0, 1) -> 0 is a fine start
};

void add_to_bin(Bin& bin, double value) {
    bin.count += 1;
    bin.sum += value;
    if (value > bin.max) bin.max = value;
}

// a value in [0, 1) falls in bin number floor(value * nbins)
int bin_of(double value, int nbins) {
    return static_cast<int>(value * nbins);
}

// critical : ONE global lock. All the updates are serialized, even two
// updates of two DIFFERENT bins, which could safely happen at the same time
void histogram_critical(int n, const double* x, int nbins, Bin* bins) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        int b = bin_of(x[i], nbins);
        #pragma omp critical
        add_to_bin(bins[b], x[i]);
    }
}

// one lock PER BIN : two threads only wait for each other when they
// update the SAME bin. A lock is an omp_lock_t that must be initialized
// before use and destroyed afterwards.
void histogram_locks(int n, const double* x, int nbins, Bin* bins) {

    omp_lock_t* locks = new omp_lock_t[nbins];
    for (int b = 0; b < nbins; ++b) omp_init_lock(&locks[b]);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        int b = bin_of(x[i], nbins);
        omp_set_lock(&locks[b]);     // wait until the lock is free, then take it
        add_to_bin(bins[b], x[i]);
        omp_unset_lock(&locks[b]);   // release it
    }

    for (int b = 0; b < nbins; ++b) omp_destroy_lock(&locks[b]);
    delete[] locks;
}

// privatization : each thread fills its OWN histogram without any
// synchronization, then merges it into the shared one ONCE at the end.
// 8 threads -> 8 merges instead of n locked updates. It is exactly what
// reduction() does under the hood for simple types (see example 1).
// Rule of thumb : the best synchronization is no synchronization.
void histogram_private(int n, const double* x, int nbins, Bin* bins) {
    #pragma omp parallel
    {
        Bin* mine = new Bin[nbins];  // declared inside the region -> private

        #pragma omp for nowait
        for (int i = 0; i < n; ++i) {
            add_to_bin(mine[bin_of(x[i], nbins)], x[i]);
        }

        #pragma omp critical
        for (int b = 0; b < nbins; ++b) {
            bins[b].count += mine[b].count;
            bins[b].sum += mine[b].sum;
            if (mine[b].max > bins[b].max) bins[b].max = mine[b].max;
        }

        delete[] mine;
    }
}

// compare two histograms field by field
bool same_histogram(int nbins, const Bin* a, const Bin* b) {
    for (int k = 0; k < nbins; ++k) {
        if (a[k].count != b[k].count || a[k].sum != b[k].sum || a[k].max != b[k].max) {
            return false;
        }
    }
    return true;
}


int main() {

    // ---- part 1 ----
    std::cout << "=== Part 1 : parallel prefix sum (single, barrier, nowait) ==="
              << std::endl;
    const int n = static_cast<int>(20e6);
    long* in = new long[n];
    long* out_seq = new long[n]();  // () -> zero-initialized, so that the
    long* out_par = new long[n]();  // page faults don't distort the timings
    for (int i = 0; i < n; ++i) in[i] = i % 7;

    double t0 = omp_get_wtime();
    prefix_sum_seq(n, in, out_seq);
    double t1 = omp_get_wtime();
    std::cout << "sequential : " << (t1 - t0) << " s" << std::endl;

    // don't expect a big speedup here : a prefix sum does almost no
    // computation per element, so its speed is limited by the memory
    // bandwidth (which a single core already uses a good part of), and
    // the parallel algorithm reads the input twice. The point of this
    // part is the synchronization, not the performance.
    t0 = omp_get_wtime();
    prefix_sum_parallel(n, in, out_par);
    t1 = omp_get_wtime();
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (out_par[i] != out_seq[i]) { ok = false; break; }
    }
    std::cout << "parallel   : " << (t1 - t0) << " s | same result as sequential ? "
              << (ok ? "yes" : "NO") << std::endl;

    delete[] in;
    delete[] out_seq;
    delete[] out_par;

    // ---- part 2 ----
    std::cout << std::endl << "=== Part 2 : master vs single ===" << std::endl;
    demo_master_single();

    // ---- part 3 ----
    std::cout << std::endl << "=== Part 3 : sections ===" << std::endl;
    const int m = static_cast<int>(10e6);
    double* x = new double[m];
    // values k / 1024 with k integer : every sum of them is exact in
    // double precision, whatever the order of the additions, so the
    // histograms below can be compared with ==
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, 1023);
    for (int i = 0; i < m; ++i) x[i] = dist(gen) / 1024.0;
    demo_sections(m, x);

    // ---- part 4 ----
    std::cout << std::endl << "=== Part 4 : ordered ===" << std::endl;
    demo_ordered();

    // ---- part 5 ----
    std::cout << std::endl << "=== Part 5 : critical vs locks vs privatization ==="
              << std::endl;
    const int nbins = 64;
    Bin* ref = new Bin[nbins];
    for (int i = 0; i < m; ++i) add_to_bin(ref[bin_of(x[i], nbins)], x[i]);

    Bin* bins = new Bin[nbins];
    t0 = omp_get_wtime();
    histogram_critical(m, x, nbins, bins);
    t1 = omp_get_wtime();
    std::cout << "critical (1 global lock) : " << (t1 - t0) << " s | correct ? "
              << (same_histogram(nbins, bins, ref) ? "yes" : "NO") << std::endl;
    delete[] bins;

    bins = new Bin[nbins];
    t0 = omp_get_wtime();
    histogram_locks(m, x, nbins, bins);
    t1 = omp_get_wtime();
    std::cout << "one lock per bin         : " << (t1 - t0) << " s | correct ? "
              << (same_histogram(nbins, bins, ref) ? "yes" : "NO") << std::endl;
    delete[] bins;

    bins = new Bin[nbins];
    t0 = omp_get_wtime();
    histogram_private(m, x, nbins, bins);
    t1 = omp_get_wtime();
    std::cout << "private copies + merge   : " << (t1 - t0) << " s | correct ? "
              << (same_histogram(nbins, bins, ref) ? "yes" : "NO") << std::endl;
    delete[] bins;

    // free the allocated memory (no leak)
    delete[] ref;
    delete[] x;
    return 0;
}
