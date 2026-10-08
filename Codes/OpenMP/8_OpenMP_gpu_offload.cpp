#include <iostream>
#include <cmath>

// initialisation of OpenMP environment variables
#include <omp.h>

// Since version 4.0, OpenMP can also OFFLOAD computations to an
// accelerator (a GPU, most of the time). It needs a compiler built with
// offloading support for your GPU :
//   NVIDIA HPC SDK : nvc++ -mp=gpu -O2 8_OpenMP_gpu_offload.cpp -o 8_OpenMP_gpu_offload
//   clang          : clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -O2 ...
//   gcc            : g++ -fopenmp -foffload=nvptx-none -O2 ...
//                    (needs the gcc-offload-nvptx package)
// With a plain "g++ -fopenmp -O2", or without any GPU, the code still
// compiles and runs : the target regions are executed on the host (CPU).
// The first lines printed by the program tell you where they really ran.
// With a recent runtime, OMP_TARGET_OFFLOAD=MANDATORY (OpenMP 5.0) also
// turns a silent fallback on the CPU into an error.
//
// The GPU has its OWN memory : the data must be copied there before the
// computation and copied back afterwards. These copies (through the PCIe
// bus) are often MUCH slower than the computation itself, so the main
// skill of GPU programming is to move as little data as possible.


// ---------------------------------------------------------------
// PART 1 : target, map and the teams / threads hierarchy
// ---------------------------------------------------------------
// SAXPY : y = a * x + y (the "hello world" of accelerators)
//
//   target                 move the execution of the region to the device
//   teams                  create a LEAGUE of teams (~ CUDA blocks : they
//                          can't synchronize with each other)
//   distribute             share the iterations among the teams
//   parallel for           share each team's iterations among its threads
//                          (~ CUDA threads of a block)
//   map(to: x[0:n])        copy x to the device before the region
//   map(tofrom: y[0:n])    copy y to the device before, and back after
//   (map(from: ...) would only copy back, map(alloc: ...) only allocate)
void saxpy(int n, float a, const float* x, float* y) {

    #pragma omp target teams distribute parallel for map(to: x[0:n]) map(tofrom: y[0:n])
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}


// ---------------------------------------------------------------
// PART 2 : a reduction on the device
// ---------------------------------------------------------------
// Same reduction clause as on the CPU (see example 1). One trap : in a
// target region, a scalar variable is FIRSTPRIVATE by default (copied to
// the device, never copied back). OpenMP 5.0 says that a reduction
// variable of a combined target construct is mapped tofrom automatically,
// but older compilers (gcc 11 for instance) follow OpenMP 4.5 and would
// silently return 0 : writing map(tofrom: sum) is portable and explicit.
double dot(int n, const float* x, const float* y) {
    double sum = 0.0;

    #pragma omp target teams distribute parallel for reduction(+:sum) \
                map(to: x[0:n], y[0:n]) map(tofrom: sum)
    for (int i = 0; i < n; ++i) {
        sum += static_cast<double>(x[i]) * y[i];
    }
    return sum;
}


// ---------------------------------------------------------------
// PART 3 : keeping the data on the device -> target data
// ---------------------------------------------------------------
// 1D heat equation, explicit scheme : at each time step
//   u_new[i] = u[i] + alpha * (u[i-1] - 2 u[i] + u[i+1])
// A typical iterative solver : many steps on the same arrays. Each loop
// iteration below does 2 time steps ("ping-pong" between the 2 arrays :
// u -> u_new, then u_new -> u), so that no copy is needed.

// "declare target" compiles a function for the device too, so that it
// can be called from a target region
#pragma omp declare target
double heat_update(double left, double center, double right, double alpha) {
    return center + alpha * (left - 2.0 * center + right);
}
#pragma omp end declare target

// NAIVE : every time step is a separate target region with its own map
// clauses -> the arrays travel to the GPU and back at EVERY step
void heat_naive(int n, int steps, double alpha, double* u, double* u_new) {
    for (int s = 0; s < steps; s += 2) {

        #pragma omp target teams distribute parallel for map(to: u[0:n]) map(tofrom: u_new[0:n])
        for (int i = 1; i < n - 1; ++i) {
            u_new[i] = heat_update(u[i - 1], u[i], u[i + 1], alpha);
        }

        #pragma omp target teams distribute parallel for map(to: u_new[0:n]) map(tofrom: u[0:n])
        for (int i = 1; i < n - 1; ++i) {
            u[i] = heat_update(u_new[i - 1], u_new[i], u_new[i + 1], alpha);
        }

        if ((s + 2) % (steps / 4) == 0) {
            std::cout << "    step " << s + 2 << " : u[10] = " << u[10] << std::endl;
        }
    }
}

// target data : the arrays are copied ONCE when entering the data region
// and ONCE when leaving it ; the target regions inside find them already
// on the device (the runtime looks them up and copies nothing).
// "target update from(...)" copies the device data back to the host on
// demand, here to monitor the solution during the run.
void heat_data_region(int n, int steps, double alpha, double* u, double* u_new) {

    #pragma omp target data map(tofrom: u[0:n], u_new[0:n])
    {
        for (int s = 0; s < steps; s += 2) {

            #pragma omp target teams distribute parallel for
            for (int i = 1; i < n - 1; ++i) {
                u_new[i] = heat_update(u[i - 1], u[i], u[i + 1], alpha);
            }

            #pragma omp target teams distribute parallel for
            for (int i = 1; i < n - 1; ++i) {
                u[i] = heat_update(u_new[i - 1], u_new[i], u_new[i + 1], alpha);
            }

            // from time to time, bring u back to the host to look at it
            if ((s + 2) % (steps / 4) == 0) {
                #pragma omp target update from(u[0:n])
                std::cout << "    step " << s + 2 << " : u[10] = " << u[10] << std::endl;
            }
        }
    }   // <- u and u_new are copied back to the host here
}


int main() {

    // ---- where do we run ? ----
    int num_devices = omp_get_num_devices();
    int on_device = 0;
    #pragma omp target map(from: on_device)
    {
        on_device = !omp_is_initial_device();
    }
    std::cout << "number of offload devices : " << num_devices << std::endl;
    std::cout << "target regions run on     : "
              << (on_device ? "the DEVICE (GPU)" : "the HOST (CPU fallback)") << std::endl;

    // ---- part 1 & 2 ----
    std::cout << std::endl << "=== Parts 1 and 2 : saxpy and dot product ===" << std::endl;
    const int n = static_cast<int>(10e6);
    float* x = new float[n];
    float* y = new float[n];
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    double t0 = omp_get_wtime();
    saxpy(n, 3.0f, x, y);  // y = 3 * 1 + 2 = 5 everywhere
    double t1 = omp_get_wtime();
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (y[i] != 5.0f) { ok = false; break; }
    }
    std::cout << "saxpy : " << (ok ? "correct" : "WRONG") << " | time = " << (t1 - t0)
              << " s (transfers included)" << std::endl;

    t0 = omp_get_wtime();
    double d = dot(n, x, y);  // 1 * 5 summed n times
    t1 = omp_get_wtime();
    std::cout << "dot   : " << d << " (expected " << 5.0 * n << ") | time = "
              << (t1 - t0) << " s" << std::endl;

    delete[] x;
    delete[] y;

    // ---- part 3 ----
    std::cout << std::endl << "=== Part 3 : target data (1D heat equation) ===" << std::endl;
    const int m = 1000000;
    const int steps = 200;  // must be a multiple of 4 (2 steps per iteration)
    const double alpha = 0.25;

    // initial temperature : hot (100) on the left end, 0 elsewhere
    double* u1 = new double[m]();
    double* w1 = new double[m]();
    double* u2 = new double[m]();
    double* w2 = new double[m]();
    u1[0] = w1[0] = u2[0] = w2[0] = 100.0;

    std::cout << "naive (map at every step) :" << std::endl;
    t0 = omp_get_wtime();
    heat_naive(m, steps, alpha, u1, w1);
    t1 = omp_get_wtime();
    std::cout << "  time = " << (t1 - t0) << " s" << std::endl;

    std::cout << "with a target data region :" << std::endl;
    t0 = omp_get_wtime();
    heat_data_region(m, steps, alpha, u2, w2);
    t1 = omp_get_wtime();
    std::cout << "  time = " << (t1 - t0) << " s" << std::endl;

    // on the CPU fallback, map clauses copy nothing (host and "device" are
    // the same memory) : both times are close. On a real GPU, the naive
    // version spends most of its time in PCIe transfers.
    double max_diff = 0.0;
    for (int i = 0; i < m; ++i) {
        max_diff = std::fmax(max_diff, std::fabs(u1[i] - u2[i]));
    }
    std::cout << "max difference between the two versions : " << max_diff << std::endl;

    // free the allocated memory (no leak)
    delete[] u1;
    delete[] w1;
    delete[] u2;
    delete[] w2;
    return 0;
}
