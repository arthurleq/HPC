#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cmath>

// nvc++ -acc -Minfo=accel -O2 6_OpenACC_heat_2D.cpp -o 6_OpenACC_heat_2D
// g++ -fopenacc -O2 6_OpenACC_heat_2D.cpp -o 6_OpenACC_heat_2D

// Heat diffusion in a square plate : EXACTLY the problem solved with MPI
// in ../MPI/7_MPI_heat_2D.cpp (a hot disk in the middle of a cold plate
// whose borders are kept at 0, explicit finite differences), this time on
// a GPU. Comparing the two codes shows the difference between the models :
//   - MPI makes the distribution of the data explicit : blocks, ghost
//     cells, messages, reductions between processes
//   - OpenACC keeps the sequential code as it is and adds directives ; the
//     whole difficulty moves to the data transfers between host and device
//
// Recipe of a good GPU port, applied here :
//   1. parallelize the loops            (parallel loop gang + loop vector)
//   2. keep the data on the device      (one data region around the time loop)
//   3. compute diagnostics on the device (reductions : only 2 numbers come back)

const int N = 512;              // the plate has N x N cells (+ a border of fixed cells)
const int W = N + 2;            // width of a row in memory, border included
const long SIZE = static_cast<long>(W) * W;
const int steps = 5000;         // number of time steps (even)
const int report_every = 1000;  // print diagnostics every ... steps
const double alpha = 0.2;       // D dt / dx^2, must be <= 0.25 for stability


double now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the update of one cell. "acc routine seq" compiles this function for the
// device too, so that the GPU kernels and the CPU reference call the SAME
// function
#pragma acc routine seq
inline double update_cell(const double* t, int i, int j) {
    return t[i * W + j] + alpha * (t[(i - 1) * W + j] + t[(i + 1) * W + j]
                                   + t[i * W + j - 1] + t[i * W + j + 1] - 4.0 * t[i * W + j]);
}

// initial state : 100 degrees inside a disk in the middle, 0 elsewhere
void initialize(double* t) {
    for (int i = 0; i < W; ++i) {
        for (int j = 0; j < W; ++j) {
            double di = i - (N + 1) / 2.0, dj = j - (N + 1) / 2.0;
            bool border = (i == 0 || j == 0 || i == N + 1 || j == N + 1);
            bool hot = di * di + dj * dj < (N / 6.0) * (N / 6.0);
            t[i * W + j] = (!border && hot) ? 100.0 : 0.0;
        }
    }
}

// one time step on the device : t -> t_new. The data is already there
// (present). The rows are shared among the gangs and the columns, which are
// contiguous in memory, among the vector lanes (example 3) : the natural
// mapping of a 2D stencil, efficient on a GPU and vectorizable on a CPU.
// (collapse(2) would work too, see the reduction in main)
void step_device(const double* t, double* t_new) {
    #pragma acc parallel loop gang present(t[0:SIZE], t_new[0:SIZE])
    for (int i = 1; i <= N; ++i) {
        #pragma acc loop vector
        for (int j = 1; j <= N; ++j) {
            t_new[i * W + j] = update_cell(t, i, j);
        }
    }
}

// the same time step on the CPU, for the reference
void step_host(const double* t, double* t_new) {
    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= N; ++j) {
            t_new[i * W + j] = update_cell(t, i, j);
        }
    }
}


int main() {

    double* t = new double[SIZE];
    double* t_new = new double[SIZE];
    initialize(t);
    initialize(t_new);  // t_new needs the border values too (never computed)

    std::cout << "plate of " << N << " x " << N << " cells, " << steps << " time steps"
              << std::endl;

    double t0 = now();

    // the two arrays live on the device during the whole simulation :
    // copied in once, t copied back once at the end
    #pragma acc data copy(t[0:SIZE]) copyin(t_new[0:SIZE])
    {
        for (int s = 0; s < steps; s += 2) {

            // two steps per iteration, "ping-pong" between the arrays, so
            // that the result is always back in t (no copy, no pointer swap)
            step_device(t, t_new);
            step_device(t_new, t);

            if ((s + 2) % report_every == 0) {
                // the diagnostics are computed ON THE DEVICE with reductions :
                // 2 numbers come back instead of the whole plate
                double heat = 0.0, max_temperature = 0.0;
                #pragma acc parallel loop collapse(2) reduction(+:heat) \
                            reduction(max:max_temperature) copy(heat, max_temperature) \
                            present(t[0:SIZE])
                for (int i = 1; i <= N; ++i) {
                    for (int j = 1; j <= N; ++j) {
                        heat += t[i * W + j];
                        max_temperature = std::fmax(max_temperature, t[i * W + j]);
                    }
                }
                std::cout << "  step " << std::setw(5) << s + 2 << " : total heat = "
                          << std::fixed << std::setprecision(0) << heat
                          << ", max temperature = " << std::setprecision(2)
                          << max_temperature << std::endl;
            }
        }
    }   // <- t is copied back to the host here

    double t_acc = now() - t0;

    // ---- CPU reference ----
    double* r = new double[SIZE];
    double* r_new = new double[SIZE];
    initialize(r);
    initialize(r_new);
    t0 = now();
    for (int s = 0; s < steps; s += 2) {
        step_host(r, r_new);
        step_host(r_new, r);
    }
    double t_cpu = now() - t0;

    double max_diff = 0.0;
    for (long k = 0; k < SIZE; ++k) {
        max_diff = std::max(max_diff, std::fabs(t[k] - r[k]));
    }
    std::cout << std::setprecision(3)
              << "OpenACC version   : " << t_acc << " s" << std::endl
              << "CPU, sequential   : " << t_cpu << " s   -> speedup " << t_cpu / t_acc
              << std::endl
              << "max difference    : " << std::scientific << std::setprecision(1)
              << max_diff << std::endl;

    // ---- a coarse picture of the plate (same as the MPI version) ----
    const char* shades = " .:-=+*#%@";
    std::cout << std::endl << "temperature map (' ' = 0 ... '@' = 100) :" << std::endl;
    for (int row = 0; row < 32; ++row) {
        std::cout << "  |";
        for (int col = 0; col < 64; ++col) {
            int gi = 1 + row * N / 32 + N / 64;
            int gj = 1 + col * N / 64 + N / 128;
            int level = static_cast<int>(t[gi * W + gj] / 100.0 * 9.999);
            std::cout << shades[std::clamp(level, 0, 9)];
        }
        std::cout << "|" << std::endl;
    }

    // free the allocated memory (no leak)
    delete[] t;
    delete[] t_new;
    delete[] r;
    delete[] r_new;
    return 0;
}
