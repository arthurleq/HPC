#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>

// nvc++ -acc -Minfo=accel -O2 2_OpenACC_data_management.cpp -o 2_OpenACC_data_management
// g++ -fopenacc -O2 2_OpenACC_data_management.cpp -o 2_OpenACC_data_management
// With nvc++, NV_ACC_NOTIFY=2 ./2_OpenACC_data_management prints every
// transfer between the host and the GPU.

// The GPU has its own memory, connected to the host by the PCIe bus
// (16-64 GB/s), much slower than the memory of the GPU itself (1-3 TB/s).
// Moving data is THE performance problem of GPU codes : a compute region
// that copies its arrays in and out at every call can easily spend more
// than 90 % of its time in transfers.
//
// The data clauses, usable on compute regions and on data regions :
//   copyin(a[0:n])    allocate on the device, copy host -> device at the start
//   copyout(a[0:n])   allocate, copy device -> host at the end
//   copy(a[0:n])      both
//   create(a[0:n])    allocate only : temporary arrays the host never needs
//   present(a[0:n])   the data is ALREADY on the device (runtime error if not)
// If an array is already present on the device, copyin / copy / create
// do nothing : the data is only moved where it enters the device.
//
// The problem : smoothing a signal, b[i] = (a[i-1] + a[i] + a[i+1]) / 3,
// many times in a row (the skeleton of every iterative method).
// Each iteration does 2 smoothings : a -> b, then b -> a.


double now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// at the ends of the signal, the missing neighbour is replaced by the
// value itself : every b[i] is written, so b needs no initial value
#pragma acc routine seq
inline float smooth(const float* a, int i, int n) {
    float left = a[std::max(i - 1, 0)];
    float right = a[std::min(i + 1, n - 1)];
    return (left + a[i] + right) / 3.0f;
}


// ---------------------------------------------------------------
// VERSION A : naive, every kernel moves its arrays
// ---------------------------------------------------------------
// Each parallel loop copies a and b to the GPU and back : 3 transfers of
// a whole array per loop, 6 per iteration.
void smooth_naive(int n, int iterations, float* a, float* b) {
    for (int it = 0; it < iterations; ++it) {

        #pragma acc parallel loop copyin(a[0:n]) copyout(b[0:n])
        for (int i = 0; i < n; ++i) b[i] = smooth(a, i, n);

        #pragma acc parallel loop copyin(b[0:n]) copyout(a[0:n])
        for (int i = 0; i < n; ++i) a[i] = smooth(b, i, n);
    }
}


// ---------------------------------------------------------------
// VERSION B : a structured data region
// ---------------------------------------------------------------
// "acc data" opens a region of the code during which the arrays live on
// the device : a is copied in at the beginning and out at the end, b is
// only allocated. The loops inside find them present : NO transfer at all
// during the iterations.
void smooth_data_region(int n, int iterations, float* a, float* b) {

    #pragma acc data copy(a[0:n]) create(b[0:n])
    {
        for (int it = 0; it < iterations; ++it) {

            #pragma acc parallel loop present(a[0:n], b[0:n])
            for (int i = 0; i < n; ++i) b[i] = smooth(a, i, n);

            #pragma acc parallel loop present(a[0:n], b[0:n])
            for (int i = 0; i < n; ++i) a[i] = smooth(b, i, n);
        }
    }
}


// ---------------------------------------------------------------
// VERSION C : unstructured data + update
// ---------------------------------------------------------------
// In a real code, the data often lives on the device across several
// functions : allocated and copied in an init function, used by the
// solver, copied back in a finalize function. A structured region (a
// block of code) can't express that : "enter data" and "exit data" can be
// placed anywhere, like new / delete.
// While the data lives on the device, "update" synchronizes the two copies
// on demand :
//   update self(a[0:n])    device -> host (also spelled "update host")
//   update device(a[0:n])  host -> device
// An update can move a PART of an array only : a[first:count].

void gpu_init(int n, float* a, float* b) {
    #pragma acc enter data copyin(a[0:n]) create(b[0:n])
}

void gpu_iterations(int n, int iterations, float* a, float* b) {
    for (int it = 0; it < iterations; ++it) {

        #pragma acc parallel loop present(a[0:n], b[0:n])
        for (int i = 0; i < n; ++i) b[i] = smooth(a, i, n);

        #pragma acc parallel loop present(a[0:n], b[0:n])
        for (int i = 0; i < n; ++i) a[i] = smooth(b, i, n);
    }
}

void gpu_finalize(int n, float* a, float* b) {
    #pragma acc exit data copyout(a[0:n]) delete(b[0:n])
}

void smooth_unstructured(int n, int iterations, float* a, float* b) {

    gpu_init(n, a, b);

    gpu_iterations(n, iterations / 2, a, b);

    // half-way : look at the value in the middle of the signal on the host
    // (bring back ONE element only), then add a "spike" there and send
    // this element back to the device
    #pragma acc update self(a[n / 2:1])
    std::cout << "    half-way, a[n/2] = " << a[n / 2] << " -> set to 1000 on the host"
              << std::endl;
    a[n / 2] = 1000.0f;
    #pragma acc update device(a[n / 2:1])

    gpu_iterations(n, iterations - iterations / 2, a, b);

    gpu_finalize(n, a, b);
}


int main() {

    const int n = 1 << 20;  // 1 M floats = 4 MB per array
    const int iterations = 100;
    const double mb = n * sizeof(float) / 1e6;

    float* a = new float[n];
    float* b = new float[n];
    float* reference = new float[n];

    // initial signal : a square wave
    auto init = [&]() {
        for (int i = 0; i < n; ++i) a[i] = ((i / 1000) % 2 == 0) ? 0.0f : 1.0f;
    };

    std::cout << "=== smoothing a signal of " << mb << " MB, " << iterations
              << " iterations ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);

    // the volumes of data moved between host and device are computed from
    // the data clauses : on a machine without GPU nothing is moved at all,
    // but this is what a GPU would have to transfer
    init();
    double t0 = now();
    smooth_naive(n, iterations, a, b);
    double t = now() - t0;
    std::copy(a, a + n, reference);
    std::cout << "A. a copy at every kernel : " << t << " s | data moved : "
              << 6.0 * iterations * mb / 1000 << " GB" << std::endl;

    init();
    t0 = now();
    smooth_data_region(n, iterations, a, b);
    t = now() - t0;
    bool same = std::equal(a, a + n, reference);
    std::cout << "B. acc data region        : " << t << " s | data moved : "
              << 2.0 * mb / 1000 << " GB | same result as A : " << (same ? "yes" : "NO")
              << std::endl;

    std::cout << "C. enter / exit data + update :" << std::endl;
    init();
    t0 = now();
    smooth_unstructured(n, iterations, a, b);
    t = now() - t0;
    std::cout << "                              " << t << " s | data moved : "
              << 2.0 * mb / 1000 << " GB + 2 floats | a[n/2] at the end = "
              << a[n / 2] << " (the spike was smoothed out)" << std::endl;

    // free the allocated memory (no leak)
    delete[] a;
    delete[] b;
    delete[] reference;
    return 0;
}
