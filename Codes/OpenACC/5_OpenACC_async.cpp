#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>

#ifdef _OPENACC
#include <openacc.h>
#endif

// nvc++ -acc -Minfo=accel -O2 5_OpenACC_async.cpp -o 5_OpenACC_async
// g++ -fopenacc -O2 5_OpenACC_async.cpp -o 5_OpenACC_async
// With nvc++, nsys profile ./5_OpenACC_async shows the timeline of the
// transfers and kernels of each queue (NVIDIA Nsight Systems).

// By default, a compute region or an update is SYNCHRONOUS : the host
// waits until it is finished. The async(q) clause makes it asynchronous :
// the operation is put in the activity QUEUE number q (a CUDA "stream")
// and the host goes on immediately.
//   - the operations of the SAME queue are executed in order
//   - the operations of DIFFERENT queues may run at the same time
//   - "#pragma acc wait(q)" waits for queue q, "#pragma acc wait" for all,
//     and wait(q) can also be a clause of another directive
//
// Two uses :
//   1. the host works while the GPU computes (part 1)
//   2. PIPELINING : a GPU has separate engines for the copies and for the
//      computations. Cut the data into chunks, and while chunk k is
//      computed, chunk k+1 is copied in and chunk k-1 is copied out :
//
//   synchronous : [ copy in everything ][ compute everything ][ copy out everything ]
//
//   pipelined   : [in 0][comp 0][out 0]
//                       [in 1][comp 1][out 1]
//                             [in 2][comp 2][out 2]          (3 queues)
//                                   [in 3][comp 3][out 3]  -> shorter in total


double now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the work done on each value : 100 iterations of the logistic map
// (see ../OpenMP/7_OpenMP_simd_collapse.cpp)
#pragma acc routine seq
inline float work(float r) {
    float v = 0.5f;
    for (int k = 0; k < 100; ++k) v = r * v * (1.0f - v);
    return v;
}


// true if the compute regions really run on an accelerator
bool on_accelerator() {
    int on_host = 1;
#ifdef _OPENACC
    #pragma acc parallel copyout(on_host)
    {
        on_host = acc_on_device(acc_device_host);
    }
#endif
    return !on_host;
}


// ---------------------------------------------------------------
// PART 1 : the host and the device work at the same time
// ---------------------------------------------------------------
// simulated host work : busy for 'seconds'
void host_work(double seconds) {
    double t0 = now();
    while (now() - t0 < seconds) { }
}

double kernel_alone(int n, const float* x, float* y) {
    double t0 = now();
    #pragma acc parallel loop present(x[0:n], y[0:n])
    for (int i = 0; i < n; ++i) y[i] = work(x[i]);
    return now() - t0;
}

double without_async(int n, const float* x, float* y, double host_seconds) {
    double t0 = now();
    #pragma acc parallel loop present(x[0:n], y[0:n])
    for (int i = 0; i < n; ++i) y[i] = work(x[i]);
    host_work(host_seconds);  // starts only when the GPU has finished
    return now() - t0;
}

double with_async(int n, const float* x, float* y, double host_seconds) {
    double t0 = now();
    #pragma acc parallel loop present(x[0:n], y[0:n]) async(1)
    for (int i = 0; i < n; ++i) y[i] = work(x[i]);
    host_work(host_seconds);  // runs WHILE the GPU computes
    #pragma acc wait(1)
    return now() - t0;
}


// ---------------------------------------------------------------
// PART 2 : pipelining transfers and computations
// ---------------------------------------------------------------
// synchronous version : copy everything, compute everything, copy back
void process_synchronous(int n, const float* x, float* y) {
    #pragma acc parallel loop copyin(x[0:n]) copyout(y[0:n])
    for (int i = 0; i < n; ++i) y[i] = work(x[i]);
}

// pipelined version : the arrays are allocated on the device once, then
// each chunk goes through its own "copy in -> compute -> copy out"
// sequence in one of 3 queues (round robin). Inside a queue the 3 steps
// are in order ; the queues overlap each other.
void process_pipelined(int n, const float* x, float* y, int nchunks) {

    #pragma acc data create(x[0:n], y[0:n])
    {
        for (int c = 0; c < nchunks; ++c) {
            int first = static_cast<int>(static_cast<long>(n) * c / nchunks);
            int count = static_cast<int>(static_cast<long>(n) * (c + 1) / nchunks) - first;
            int queue = 1 + c % 3;

            #pragma acc update device(x[first:count]) async(queue)

            #pragma acc parallel loop present(x[0:n], y[0:n]) async(queue)
            for (int i = first; i < first + count; ++i) y[i] = work(x[i]);

            #pragma acc update self(y[first:count]) async(queue)
        }

        // the host must not leave the data region (which frees the device
        // arrays) nor read y before everything is finished
        #pragma acc wait
    }
}


int main() {

    // a GPU needs a big problem to show anything ; without GPU (the CPU
    // fallback of gcc uses a single thread), a small one runs in seconds
    const bool gpu = on_accelerator();
    const int n = gpu ? (1 << 25) : (1 << 20);  // 32 M or 1 M values
    std::cout << "running on " << (gpu ? "an accelerator" : "the host (CPU fallback)")
              << ", n = " << n << std::endl << std::endl;

    float* x = new float[n];
    float* y = new float[n];
    float* reference = new float[n];
    for (int i = 0; i < n; ++i) x[i] = 2.5f + 1.5f * i / n;

    std::cout << std::fixed << std::setprecision(3);

    // ---- part 1 ----
    std::cout << "=== Part 1 : host and device working together ===" << std::endl;
    #pragma acc data copyin(x[0:n]) create(y[0:n])
    {
        // the host gets as much work as the device, to make the effect visible
        double t_kernel = kernel_alone(n, x, y);
        double t_sync = without_async(n, x, y, t_kernel);
        double t_async = with_async(n, x, y, t_kernel);
        std::cout << "kernel alone                     : " << t_kernel << " s" << std::endl
                  << "kernel, then as much host work   : " << t_sync << " s" << std::endl
                  << "both at the same time (async)    : " << t_async << " s" << std::endl;
    }

    // ---- part 2 ----
    std::cout << std::endl << "=== Part 2 : pipelining (" << n * sizeof(float) / 1e6
              << " MB in, the same out) ===" << std::endl;

    double t0 = now();
    process_synchronous(n, x, reference);
    double t_sync = now() - t0;
    std::cout << "synchronous          : " << t_sync << " s" << std::endl;

    for (int nchunks : {4, 16}) {
        std::fill(y, y + n, 0.0f);
        t0 = now();
        process_pipelined(n, x, y, nchunks);
        double t = now() - t0;
        bool same = std::equal(y, y + n, reference);
        std::cout << "pipelined, " << std::setw(2) << nchunks << " chunks : " << t
                  << " s | same result : " << (same ? "yes" : "NO") << std::endl;
    }

    // free the allocated memory (no leak)
    delete[] x;
    delete[] y;
    delete[] reference;
    return 0;
}
