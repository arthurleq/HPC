#include <iostream>
#include <iomanip>
#include <cmath>

// initialisation of OpenMP environment variables
#include <omp.h>

// g++ -fopenmp -O2 -march=native 7_OpenMP_simd_collapse.cpp -o 7_OpenMP_simd_collapse
// ./7_OpenMP_simd_collapse
// (-march=native lets the compiler use the widest vector instructions of
// YOUR processor, e.g. AVX2 or AVX-512, instead of the old SSE2 baseline)


// ---------------------------------------------------------------
// PART 1 : simd -> vectorizing a reduction
// ---------------------------------------------------------------
// Every modern core can apply the same operation to several numbers with
// ONE instruction : this is SIMD (Single Instruction Multiple Data, see
// Flynn's taxonomy). A 256-bit AVX2 register holds 4 doubles, a 512-bit
// AVX-512 register holds 8 :
//
//   scalar :  s += x[0]*y[0]   then   s += x[1]*y[1]   then ...
//   simd   :  [s0 s1 s2 s3] += [x0 x1 x2 x3] * [y0 y1 y2 y3]   (1 instruction)
//
// Compilers vectorize loops by themselves (gcc : from -O3, or -O2 since
// gcc 12), but only when they can prove it doesn't change the result.
// "#pragma omp simd" tells the compiler : vectorize this loop, I take the
// responsibility. It is a different level of parallelism from threads :
// it works INSIDE one core, and can be combined with threads (part 2).

// plain loop : with gcc -O2 it is NOT vectorized. Even at -O3 it would
// not be, because a floating-point sum can't be reordered without
// changing the result : (a + b) + c is not exactly a + (b + c) in floating
// point, and the vectorized version adds the numbers in a different order.
double dot_scalar(int n, const double* x, const double* y) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += x[i] * y[i];
    }
    return s;
}

// reduction(+:s) on a simd loop : each vector lane accumulates its own
// partial sum, and the lanes are added together at the end. We explicitly
// accept that the additions happen in another order.
double dot_simd(int n, const double* x, const double* y) {
    double s = 0.0;
    #pragma omp simd reduction(+:s)
    for (int i = 0; i < n; ++i) {
        s += x[i] * y[i];
    }
    return s;
}


// ---------------------------------------------------------------
// PART 2 : threads AND simd -> parallel for simd
// ---------------------------------------------------------------
// The logistic map v -> r v (1 - v) is the most famous example of chaos :
// iterated from v = 0.5, it converges to a fixed value, oscillates
// between 2, 4, 8... values, or behaves chaotically depending on r (this
// is how its "bifurcation diagram" is drawn). For each value of r, we
// iterate it 200 times : 600 floating-point operations per number read
// from memory, so the loop is limited by the computation, the best case
// for both threads and SIMD.
// "parallel for simd" first splits the iterations among the threads
// (MIMD), then each thread vectorizes its own chunk (SIMD).

// "declare simd" asks the compiler to ALSO generate a vector version of
// the function, which processes 4 or 8 values of r at once. It is what a
// simd loop calls when the function can't be inlined (e.g. when it is
// defined in another file) ; here the call is inlined anyway.
#pragma omp declare simd
inline double logistic(double r) {
    double v = 0.5;
    for (int k = 0; k < 200; ++k) {
        v = r * v * (1.0 - v);
    }
    return v;
}

void logistic_scalar(int n, const double* r, double* v) {
    for (int i = 0; i < n; ++i) v[i] = logistic(r[i]);
}

void logistic_simd(int n, const double* r, double* v) {
    #pragma omp simd
    for (int i = 0; i < n; ++i) v[i] = logistic(r[i]);
}

void logistic_parallel(int n, const double* r, double* v) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) v[i] = logistic(r[i]);
}

void logistic_parallel_simd(int n, const double* r, double* v) {
    #pragma omp parallel for simd
    for (int i = 0; i < n; ++i) v[i] = logistic(r[i]);
}


// ---------------------------------------------------------------
// PART 3 : collapse -> sharing nested loops
// ---------------------------------------------------------------
// An RGB image stored channel by channel : img[c * npix + p] with c = 0,
// 1, 2 (red, green, blue) and p = 0 .. npix-1. We apply a gamma
// correction to every value.
// "parallel for" only shares the OUTER loop : 3 iterations to give to
// 8 threads -> 3 threads work, the other 5 wait. collapse(2) merges the
// two loops into ONE loop of 3 * npix iterations, shared among all the
// threads. The loops must be perfectly nested (no code between the two
// "for") and their bounds must not depend on each other.
//
// "parallel for" is a shortcut for "parallel" (create the team) followed
// by "for" (share the loop) ; they are written separately here so that
// each thread can count the iterations it received.

// returns the number of threads that actually got some work
int gamma_outer_only(int nchan, int npix, const double* img, double* out) {
    int busy = 0;

    #pragma omp parallel
    {
        long my_iterations = 0;  // declared inside the region -> private

        #pragma omp for
        for (int c = 0; c < nchan; ++c) {
            for (int p = 0; p < npix; ++p) {
                out[c * npix + p] = std::pow(img[c * npix + p], 1.0 / 2.2);
                ++my_iterations;
            }
        }

        if (my_iterations > 0) {
            #pragma omp atomic
            ++busy;
        }
    }
    return busy;
}

int gamma_collapse(int nchan, int npix, const double* img, double* out) {
    int busy = 0;

    #pragma omp parallel
    {
        long my_iterations = 0;

        #pragma omp for collapse(2)
        for (int c = 0; c < nchan; ++c) {
            for (int p = 0; p < npix; ++p) {
                out[c * npix + p] = std::pow(img[c * npix + p], 1.0 / 2.2);
                ++my_iterations;
            }
        }

        if (my_iterations > 0) {
            #pragma omp atomic
            ++busy;
        }
    }
    return busy;
}


int main() {

    // the first parallel region creates the threads : do it once here so
    // that this cost does not pollute the timings below
    #pragma omp parallel
    { }

    std::cout << "=== Part 1 : simd reduction (dot product, 1 thread) ===" << std::endl;

    // small arrays (2 x 32 KB) that stay in the cache, used many times :
    // the loop is limited by the computation, not by the memory
    const int n = 4096;
    const int repeat = 200000;
    double* x = new double[n];
    double* y = new double[n];
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0 / (i + 1);
        y[i] = std::cos(i);
    }

    double s_scalar = 0.0, s_simd = 0.0;
    double t0 = omp_get_wtime();
    for (int r = 0; r < repeat; ++r) s_scalar = dot_scalar(n, x, y);
    double t_scalar = omp_get_wtime() - t0;

    t0 = omp_get_wtime();
    for (int r = 0; r < repeat; ++r) s_simd = dot_simd(n, x, y);
    double t_simd = omp_get_wtime() - t0;

    std::cout << std::setprecision(3);
    std::cout << "scalar : " << t_scalar << " s | dot = " << std::setprecision(17)
              << s_scalar << std::endl;
    std::cout << std::setprecision(3);
    std::cout << "simd   : " << t_simd << " s | dot = " << std::setprecision(17)
              << s_simd << std::endl;
    std::cout << std::setprecision(3);
    std::cout << "speedup of simd : " << t_scalar / t_simd
              << " | difference of the results : " << s_simd - s_scalar
              << " (other order of the additions)" << std::endl;

    delete[] x;
    delete[] y;

    std::cout << std::endl << "=== Part 2 : threads and simd (logistic map) ==="
              << std::endl;

    const int m = static_cast<int>(1e6);
    double* r = new double[m];
    double* v = new double[m];
    double* v_ref = new double[m];
    for (int i = 0; i < m; ++i) {
        r[i] = 2.5 + 1.5 * i / m;  // r from 2.5 to 4
        v[i] = 0.0;
        v_ref[i] = 0.0;
    }

    t0 = omp_get_wtime();
    logistic_scalar(m, r, v_ref);
    double t_ref = omp_get_wtime() - t0;
    std::cout << "1 thread , scalar : " << t_ref << " s" << std::endl;

    // every version computes each value with exactly the same operations,
    // in the same order (no reduction here) : results must be identical
    auto check = [&]() {
        for (int i = 0; i < m; ++i) {
            if (v[i] != v_ref[i]) return "  <- DIFFERENT";
        }
        return "";
    };

    t0 = omp_get_wtime();
    logistic_simd(m, r, v);
    double t = omp_get_wtime() - t0;
    std::cout << "1 thread , simd   : " << t << " s | speedup " << t_ref / t
              << check() << std::endl;

    t0 = omp_get_wtime();
    logistic_parallel(m, r, v);
    t = omp_get_wtime() - t0;
    std::cout << omp_get_max_threads() << " threads, scalar : " << t
              << " s | speedup " << t_ref / t << check() << std::endl;

    t0 = omp_get_wtime();
    logistic_parallel_simd(m, r, v);
    t = omp_get_wtime() - t0;
    std::cout << omp_get_max_threads() << " threads, simd   : " << t
              << " s | speedup " << t_ref / t << check() << std::endl;

    delete[] r;
    delete[] v;
    delete[] v_ref;

    std::cout << std::endl << "=== Part 3 : collapse (gamma correction of an RGB image) ==="
              << std::endl;

    const int nchan = 3;
    const int npix = static_cast<int>(4e6);
    double* img = new double[nchan * npix];
    double* out = new double[nchan * npix];
    for (int i = 0; i < nchan * npix; ++i) {
        img[i] = (i % 256) / 255.0;
        out[i] = 0.0;
    }

    t0 = omp_get_wtime();
    int busy = gamma_outer_only(nchan, npix, img, out);
    t = omp_get_wtime() - t0;
    std::cout << "parallel for (outer loop only) : " << t << " s | threads with work : "
              << busy << " / " << omp_get_max_threads() << std::endl;

    t0 = omp_get_wtime();
    busy = gamma_collapse(nchan, npix, img, out);
    t = omp_get_wtime() - t0;
    std::cout << "parallel for collapse(2)       : " << t << " s | threads with work : "
              << busy << " / " << omp_get_max_threads() << std::endl;

    // free the allocated memory (no leak)
    delete[] img;
    delete[] out;
    return 0;
}
