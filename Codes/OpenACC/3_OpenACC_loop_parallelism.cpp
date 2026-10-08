#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cstdlib>

// nvc++ -acc -Minfo=accel -O2 3_OpenACC_loop_parallelism.cpp -o 3_OpenACC_loop_parallelism
// g++ -fopenacc -O2 3_OpenACC_loop_parallelism.cpp -o 3_OpenACC_loop_parallelism
// ./3_OpenACC_loop_parallelism [N]      (N x N matrices, default 512 ; try 4096 on a GPU)

// OpenACC exposes THREE levels of parallelism, that the compiler maps on
// the hardware :
//
//   level    granularity    can synchronize ?        NVIDIA GPU            multicore CPU
//   gang     coarse         no (independent)         thread block          thread
//   worker   medium         yes, inside a gang       warp (32 threads)     -
//   vector   fine (SIMT)    yes, inside a worker     threads of a warp     SIMD lanes
//
// By default the compiler chooses the mapping itself (a "parallel loop"
// usually becomes "gang vector"). The loop clauses give more control :
//   gang / worker / vector / seq   which level(s) parallelize this loop
//   collapse(n)                    merge n nested loops into one
//   tile(a, b)                     cut a 2D loop nest into tiles of a x b
// and on the parallel construct : num_gangs(n), num_workers(n),
// vector_length(n) set the sizes. Variables declared inside a loop are
// private ; private(x) privatizes a variable declared outside.
//
// Test case : the matrix product C = A x B (N x N matrices stored row by
// row). Each C[i][j] is the dot product of row i of A and column j of B :
// N^3 multiply-adds, the most studied kernel of numerical computing.


double now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}


// VERSION 1 : collapse(2). The i and j loops become ONE loop of N^2
// iterations shared among gangs and vector lanes ; each iteration computes
// one C[i][j] with a sequential k loop. Simple and often good enough.
void matmul_collapse(int N, const float* A, const float* B, float* C) {
    #pragma acc parallel loop collapse(2) present(A[0:N*N], B[0:N*N], C[0:N*N])
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;  // declared inside the loop : private
            #pragma acc loop seq
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// VERSION 2 : explicit mapping, rows -> gangs, columns -> vector lanes.
// The lanes of a warp take CONSECUTIVE values of j : at each k they read
// consecutive elements B[k*N + j] and write consecutive C[i*N + j]. On a
// GPU, such accesses are "coalesced" into a few wide memory transactions
// (see ../CUDA/4_CUDA_memory_coalescing.cu).
void matmul_gang_vector(int N, const float* A, const float* B, float* C) {
    #pragma acc parallel loop gang vector_length(128) present(A[0:N*N], B[0:N*N], C[0:N*N])
    for (int i = 0; i < N; ++i) {
        #pragma acc loop vector
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// VERSION 3 : the SAME mapping upside down, columns -> gangs, rows ->
// vector lanes. Now consecutive lanes take consecutive i : they read
// A[i*N + k] and write C[i*N + j] N elements apart. Each lane needs its own
// memory transaction : on a GPU this version is MUCH slower. The vector
// level must always run along the contiguous dimension of the arrays.
void matmul_bad_mapping(int N, const float* A, const float* B, float* C) {
    #pragma acc parallel loop gang vector_length(128) present(A[0:N*N], B[0:N*N], C[0:N*N])
    for (int j = 0; j < N; ++j) {
        #pragma acc loop vector
        for (int i = 0; i < N; ++i) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// VERSION 4 : tile(32, 32). The (i, j) space is cut into tiles of 32 x 32
// elements, each tile being processed by one gang : the rows of A and the
// columns of B needed by a tile are shared by its 1024 elements and get
// reused from the cache instead of being read again from memory.
void matmul_tile(int N, const float* A, const float* B, float* C) {
    #pragma acc parallel loop tile(32, 32) present(A[0:N*N], B[0:N*N], C[0:N*N])
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


int main(int argc, char** argv) {

    const int N = (argc > 1) ? std::atoi(argv[1]) : 512;
    const long NN = static_cast<long>(N) * N;

    float* A = new float[NN];
    float* B = new float[NN];
    float* C = new float[NN];
    float* reference = new float[NN]();

    for (long i = 0; i < NN; ++i) {
        A[i] = static_cast<float>(i % 7) - 3.0f;
        B[i] = static_cast<float>(i % 5) * 0.5f;
    }

    // CPU reference, with the cache-friendly loop order i, k, j
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < N; ++j)
                reference[i * N + j] += A[i * N + k] * B[k * N + j];

    std::cout << "C = A x B with N = " << N << " (" << 2.0 * N * N * N / 1e9
              << " GFLOP per product)" << std::endl;

    // the matrices are copied ONCE to the device for all the versions (see
    // example 2) ; C is brought back after each version to be checked
    #pragma acc data copyin(A[0:NN], B[0:NN]) create(C[0:NN])
    {
        auto run = [&](const char* name, void (*matmul)(int, const float*, const float*, float*)) {
            double t0 = now();
            matmul(N, A, B, C);
            double t = now() - t0;

            #pragma acc update self(C[0:NN])
            double max_error = 0.0;
            for (long i = 0; i < NN; ++i) {
                double scale = std::fmax(1.0, std::fabs(reference[i]));
                max_error = std::fmax(max_error, std::fabs(C[i] - reference[i]) / scale);
            }
            std::cout << name << std::fixed << std::setprecision(3) << t << " s | "
                      << std::setw(7) << std::setprecision(2) << 2.0 * N * N * N / t / 1e9
                      << " GFLOP/s | max relative error " << std::scientific
                      << std::setprecision(1) << max_error << std::endl;
        };

        run("1. collapse(2)                 : ", matmul_collapse);
        run("2. gang (rows) + vector (cols) : ", matmul_gang_vector);
        run("3. gang (cols) + vector (rows) : ", matmul_bad_mapping);
        run("4. tile(32, 32)                : ", matmul_tile);
    }

    // free the allocated memory (no leak)
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] reference;
    return 0;
}
