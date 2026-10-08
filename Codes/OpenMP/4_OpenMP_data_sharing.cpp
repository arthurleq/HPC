#include <iostream>
#include <cmath>

// initialisation of OpenMP environment variables
#include <omp.h>

// g++ -fopenmp -O2 4_OpenMP_data_sharing.cpp -o 4_OpenMP_data_sharing
// ./4_OpenMP_data_sharing

// In a parallel region, every variable is either SHARED (one single
// copy, seen by every thread) or PRIVATE (one copy per thread). Getting
// this right is THE most important skill in OpenMP : most OpenMP bugs
// are a variable that is shared when it should be private.
// 
// The default rules :
//   - variables declared OUTSIDE the region            -> shared
//   - variables declared INSIDE the region             -> private
//   - the index of a loop shared with "omp for"        -> private
// and the clauses that change them :
//   shared(x)       one copy for everybody (the default, see above)
//   private(x)      one NEW, UNINITIALIZED copy per thread
//   firstprivate(x) one copy per thread, initialized with the value x
//                   had just before the region
//   lastprivate(x)  private, and after the loop x receives the value of
//                   the sequentially LAST iteration
//   default(none)   no default rule at all : every variable used in the
//                   region must be listed explicitly
//   reduction(op:x) private copies combined at the end (see example 1)


// ---------------------------------------------------------------
// PART 1 : private / firstprivate / lastprivate
// ---------------------------------------------------------------

// private : each thread gets its own copy of x, which is NOT initialized
// (reading it before writing it is a bug). The original x is untouched
// by the region and keeps its value afterwards.
void demo_private() {
    int x = 42;

    #pragma omp parallel private(x) num_threads(4)
    {
        x = omp_get_thread_num();   // each thread writes its OWN x
        #pragma omp critical
        std::cout << "  [private]      thread " << omp_get_thread_num()
                  << " : its x = " << x << std::endl;
    }

    std::cout << "  [private]      after the region, x = " << x
              << " (unchanged)" << std::endl;
}

// firstprivate : same as private, but every copy starts with the value
// of x just before the region. Typical use : a parameter that each
// thread modifies locally (a seed, an offset, a counter...).
void demo_firstprivate() {
    int x = 42;

    #pragma omp parallel firstprivate(x) num_threads(4)
    {
        x += omp_get_thread_num();  // each copy starts at 42
        #pragma omp critical
        std::cout << "  [firstprivate] thread " << omp_get_thread_num()
                  << " : its x = " << x << std::endl;
    }

    std::cout << "  [firstprivate] after the region, x = " << x
              << " (unchanged)" << std::endl;
}

// lastprivate : after the loop, x holds the value it had at the end of
// the iteration that would have been executed LAST by a sequential
// program (i = n-1), whatever thread actually executed it.
void demo_lastprivate() {
    const int n = 100;
    double x = 0.0;

    #pragma omp parallel for lastprivate(x)
    for (int i = 0; i < n; ++i) {
        x = std::sqrt(static_cast<double>(i));
    }

    std::cout << "  [lastprivate]  after the loop, x = " << x
              << " = sqrt(" << n - 1 << ")" << std::endl;
}


// ---------------------------------------------------------------
// PART 2 : a real-life bug -> a shared scratch buffer
// ---------------------------------------------------------------
// Median filter : out[i] = median of the 5 values in[i-2] .. in[i+2].
// It removes "spikes" from a noisy signal much better than an average.
// To find the median, the 5 values are copied into a small scratch
// buffer (the "window") and sorted.

// median of the 5 values of w (sorts w in place with an insertion sort)
double median_of_5(double* w) {
    for (int a = 1; a < 5; ++a) {
        double v = w[a];
        int b = a - 1;
        while (b >= 0 && w[b] > v) {
            w[b + 1] = w[b];
            --b;
        }
        w[b + 1] = v;
    }
    return w[2];
}

// sequential reference
void median_filter_seq(int n, const double* in, double* out) {
    double window[5];
    for (int i = 2; i < n - 2; ++i) {
        for (int k = 0; k < 5; ++k) window[k] = in[i - 2 + k];
        out[i] = median_of_5(window);
    }
}

// BUGGY parallel version : exactly the sequential code plus one pragma.
// But window is declared OUTSIDE the parallel region -> it is SHARED :
// all threads copy their values into the SAME 5 cells and sort them at
// the same time, overwriting each other's data. This is a data race :
// the result is wrong, and different at every run.
// Bonus : it is also SLOWER than the correct version, because every
// thread keeps writing to the same cache line, which must then bounce
// from core to core (cache coherence traffic).
void median_filter_shared_bug(int n, const double* in, double* out) {
    double window[5];

    #pragma omp parallel for
    for (int i = 2; i < n - 2; ++i) {
        for (int k = 0; k < 5; ++k) window[k] = in[i - 2 + k];
        out[i] = median_of_5(window);
    }
}

// FIX 1 : privatize the buffer with a clause. A private array is a whole
// array per thread (5 doubles each here).
void median_filter_private(int n, const double* in, double* out) {
    double window[5];

    #pragma omp parallel for private(window)
    for (int i = 2; i < n - 2; ++i) {
        for (int k = 0; k < 5; ++k) window[k] = in[i - 2 + k];
        out[i] = median_of_5(window);
    }
}

// FIX 2 (the most readable one) : declare variables in the smallest
// possible scope. Declared inside the loop body, window is automatically
// private. default(none) forces us to state the status of every variable
// used in the region : forgetting one is a COMPILE error instead of a
// silent race. For instance, removing "in" from the shared() list gives :
//   error: 'in' not specified in enclosing 'parallel'
void median_filter_default_none(int n, const double* in, double* out) {

    #pragma omp parallel for default(none) shared(n, in, out)
    for (int i = 2; i < n - 2; ++i) {
        double window[5];
        for (int k = 0; k < 5; ++k) window[k] = in[i - 2 + k];
        out[i] = median_of_5(window);
    }
}

// count the cells of out that differ from the reference
long count_errors(int n, const double* out, const double* ref) {
    long errors = 0;
    for (int i = 2; i < n - 2; ++i) {
        if (out[i] != ref[i]) ++errors;
    }
    return errors;
}


int main() {

    std::cout << "=== Part 1 : private / firstprivate / lastprivate ===" << std::endl;
    demo_private();
    demo_firstprivate();
    demo_lastprivate();

    std::cout << std::endl << "=== Part 2 : shared scratch buffer bug (median filter) ==="
              << std::endl;

    const int n = static_cast<int>(4e6);
    double* in = new double[n];
    double* ref = new double[n]();   // () -> zero-initialized
    double* out = new double[n]();

    // a smooth signal + a "spike" every 13 points
    for (int i = 0; i < n; ++i) {
        in[i] = std::sin(0.001 * i) + (i % 13 == 0 ? 5.0 : 0.0);
    }

    median_filter_seq(n, in, ref);

    double t0 = omp_get_wtime();
    median_filter_shared_bug(n, in, out);
    double t1 = omp_get_wtime();
    std::cout << "shared window (BUG)   : " << count_errors(n, out, ref)
              << " wrong values | time = " << (t1 - t0) << " s" << std::endl;

    t0 = omp_get_wtime();
    median_filter_private(n, in, out);
    t1 = omp_get_wtime();
    std::cout << "private(window)       : " << count_errors(n, out, ref)
              << " wrong values | time = " << (t1 - t0) << " s" << std::endl;

    t0 = omp_get_wtime();
    median_filter_default_none(n, in, out);
    t1 = omp_get_wtime();
    std::cout << "default(none), local  : " << count_errors(n, out, ref)
              << " wrong values | time = " << (t1 - t0) << " s" << std::endl;

    // free the allocated memory (no leak)
    delete[] in;
    delete[] ref;
    delete[] out;
    return 0;
}
