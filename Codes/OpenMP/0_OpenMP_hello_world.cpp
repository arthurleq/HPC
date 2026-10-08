#include <iostream>
#include <iomanip>
#include <cmath>

// initialisation of OpenMP environment variables
#include <omp.h>

// g++ -fopenmp -O2 0_OpenMP_hello_world.cpp -o 0_OpenMP_hello_world
// ./0_OpenMP_hello_world
// OMP_NUM_THREADS=2 ./0_OpenMP_hello_world


// ---------------------------------------------------------------
// PART 1 : the fork-join model
// ---------------------------------------------------------------
// A program always starts with ONE thread (the "master", or "initial"
// thread). When it reaches a "#pragma omp parallel" region, it FORKS a
// team of threads which ALL execute the block below. At the end of the
// block, every thread waits for the others (implicit barrier) and the
// team JOINS back into the master thread alone.
//
//                 fork                        join
//   master ---------+---> thread 0 ------------+---> master
//                   +---> thread 1 ------------+
//                   +---> thread 2 ------------+
//                   +---> thread 3 ------------+
void fork_join() {

    std::cout << "before the parallel region : " << omp_get_num_threads()
              << " thread" << std::endl;

    #pragma omp parallel
    {
        // the variables declared INSIDE the region are private : each
        // thread has its own copy of tid and nth
        int tid = omp_get_thread_num();   // my id inside the team : 0 .. nth-1
        int nth = omp_get_num_threads();  // size of the team

        // std::cout is shared by every thread : without protection, the
        // characters of two lines printed at the same time can get mixed.
        // Note that the ORDER of the lines still changes from one run to
        // the other : threads run concurrently, nobody is "first".
        #pragma omp critical
        std::cout << "  hello from thread " << tid << " / " << nth << std::endl;

    }   // <- implicit barrier, then join

    std::cout << "after the parallel region  : " << omp_get_num_threads()
              << " thread" << std::endl;
}


// ---------------------------------------------------------------
// PART 2 : choosing the number of threads
// ---------------------------------------------------------------
// From the lowest to the highest priority :
//   1. the environment variable : OMP_NUM_THREADS=4 ./0_OpenMP_hello_world
//   2. the runtime function     : omp_set_num_threads(4);
//   3. the clause               : #pragma omp parallel num_threads(4)
// Without any of them, the runtime usually creates one thread per
// logical core (hyper-threads included).
void choose_num_threads() {

    // number of logical cores the program is allowed to run on
    std::cout << "omp_get_num_procs()   = " << omp_get_num_procs() << std::endl;
    // number of threads the NEXT parallel region will get by default
    std::cout << "omp_get_max_threads() = " << omp_get_max_threads() << std::endl;

    // the clause wins over everything else, but only for this region
    #pragma omp parallel num_threads(3)
    {
        if (omp_get_thread_num() == 0) {
            std::cout << "  num_threads(3)    -> team of " << omp_get_num_threads()
                      << " threads" << std::endl;
        }
    }

    // the if clause : the region is executed by a team only if the
    // condition is true, otherwise by the master thread alone. Useful
    // not to pay the fork-join overhead (a few microseconds) for a
    // problem too small to be worth it.
    int n = 100;
    #pragma omp parallel if(n > 10000)
    {
        if (omp_get_thread_num() == 0) {
            std::cout << "  if(n > 10000)     -> team of " << omp_get_num_threads()
                      << " thread (n = " << n << ")" << std::endl;
        }
    }
}


// ---------------------------------------------------------------
// PART 3 : a first parallel loop, and how much faster it runs
// ---------------------------------------------------------------
// "#pragma omp parallel for" = fork a team AND share the iterations of
// the loop among its threads (each thread gets about n / P of them).
// It is only correct because the iterations are INDEPENDENT : iteration
// i only writes y[i] and only reads x[i], so they can run in any order.
// The loop index i is automatically private to each thread, while x, y
// and n, declared outside the region, are shared.
void compute(long n, const double* x, double* y) {

    #pragma omp parallel for

    for (long i = 0; i < n; ++i) {
        // a deliberately "expensive" body (transcendental functions), so
        // that the loop is limited by the computation and not by the
        // memory bandwidth -> the speedup is easy to see
        y[i] = std::sin(x[i]) * std::exp(-x[i] * x[i]) + std::sqrt(x[i]);
    }
}

// time compute() with p threads. A timing on a laptop is noisy (other
// programs, turbo frequency...) so we keep the best of a few runs, which
// is the usual way to benchmark a short kernel.
double time_compute(int p, long n, const double* x, double* y) {
    omp_set_num_threads(p);
    double best = 1e30;
    for (int run = 0; run < 3; ++run) {
        double t0 = omp_get_wtime();
        compute(n, x, y);
        double t = omp_get_wtime() - t0;
        if (t < best) best = t;
    }
    return best;
}


int main() {

#ifdef _OPENMP
    // _OPENMP is defined by the compiler only when -fopenmp is used. Its
    // value is the release date (yyyymm) of the supported specification,
    // e.g. 201511 = OpenMP 4.5, 201811 = 5.0, 202011 = 5.1
    std::cout << "compiled with OpenMP, _OPENMP = " << _OPENMP << std::endl;
#endif

    std::cout << std::endl << "=== Part 1 : fork-join ===" << std::endl;
    fork_join();

    std::cout << std::endl << "=== Part 2 : number of threads ===" << std::endl;
    choose_num_threads();

    std::cout << std::endl << "=== Part 3 : speedup of a parallel loop ===" << std::endl;

    const long n = static_cast<long>(10e6);
    double* x = new double[n];
    double* y = new double[n];
    double* y_ref = new double[n];

    // writing the arrays once BEFORE timing anything : the very first
    // write to a page of freshly allocated memory triggers a page fault
    // (the OS maps the page only then), which would unfairly slow down
    // the first measurement
    for (long i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i) / n;
        y[i] = 0.0;
        y_ref[i] = 0.0;
    }

    // reference run with a single thread : T(1)
    double t_1 = time_compute(1, n, x, y_ref);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "threads | time (s) | speedup S = T(1)/T(P) | efficiency S/P" << std::endl;

    // double the number of threads up to the number of logical cores
    const int max_threads = omp_get_num_procs();
    for (int p = 1; p <= max_threads; p *= 2) {

        double t_p = (p == 1) ? t_1 : time_compute(p, n, x, y);

        // every thread count must give exactly the same values : the
        // work is split differently, but each y[i] is computed the same way
        bool same = true;
        if (p > 1) {
            for (long i = 0; i < n; ++i) {
                if (y[i] != y_ref[i]) { same = false; break; }
            }
        }

        double speedup = t_1 / t_p;
        std::cout << std::setw(5) << p << "   | " << std::setw(8) << t_p
                  << " | " << std::setw(15) << speedup << "       | "
                  << std::setw(10) << speedup / p
                  << (same ? "" : "   <- WRONG RESULT") << std::endl;
    }

    // free the allocated memory (no leak)
    delete[] x;
    delete[] y;
    delete[] y_ref;
    return 0;
}
