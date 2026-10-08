#include <iostream>
#include <cmath>
#include <chrono>

#ifdef _OPENACC
#include <openacc.h>
#endif

// NVIDIA GPU (NVIDIA HPC SDK, the reference OpenACC compiler) :
//   nvc++ -acc -Minfo=accel -O2 1_OpenACC_first_steps.cpp -o 1_OpenACC_first_steps
// multicore CPU, same code :
//   nvc++ -acc=multicore -Minfo=accel -O2 1_OpenACC_first_steps.cpp -o 1_OpenACC_first_steps
// GCC (GPU if gcc was built with offloading support, CPU otherwise) :
//   g++ -fopenacc -O2 1_OpenACC_first_steps.cpp -o 1_OpenACC_first_steps
// -Minfo=accel makes nvc++ explain what it did with each loop : READ IT.

// OpenACC is a set of directives to offload loops to an accelerator
// (a GPU, most of the time). Same philosophy as OpenMP : you annotate a
// sequential code, and without -acc the pragmas are simply ignored.
// The difference is in the spirit : OpenACC is DESCRIPTIVE (you describe
// which loops are parallel, the compiler decides how to map them on the
// hardware), where OpenMP is PRESCRIPTIVE (you say exactly what to do).
//
// The GPU has its own memory : the data clauses say what to copy
//   copyin(x[0:n])   host -> device before the region
//   copyout(y[0:n])  device -> host after the region
//   copy(y[0:n])     both
// (x[0:n] = n elements starting at x[0] : a pointer has no size, the
// compiler must be told how much to copy). Example 2 goes into details.


// simple wall-clock timer
double now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}


// ---------------------------------------------------------------
// PART 1 : "kernels" vs "parallel loop" on SAXPY (y = a x + y)
// ---------------------------------------------------------------

// kernels : the compiler analyzes the region and only parallelizes what
// it can PROVE to be safe. Here it can't : x and y are plain pointers, and
// nothing tells it that they don't overlap (if y == x + 1, iteration i
// would write what iteration i+1 reads). It then keeps the loop
// SEQUENTIAL, which on a GPU means a single thread for the whole loop :
// with nvc++, -Minfo=accel reports a "loop carried dependence".
void saxpy_kernels(int n, float a, const float* x, float* y) {
    #pragma acc kernels copyin(x[0:n]) copy(y[0:n])
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

// kernels + "loop independent" : the programmer guarantees that the
// iterations are independent. (Declaring the pointers __restrict__, i.e.
// "they don't overlap", would also do the job.)
void saxpy_kernels_independent(int n, float a, const float* x, float* y) {
    #pragma acc kernels copyin(x[0:n]) copy(y[0:n])
    {
        #pragma acc loop independent
        for (int i = 0; i < n; ++i) {
            y[i] = a * x[i] + y[i];
        }
    }
}

// parallel loop : the programmer ASSERTS that the loop is parallel (like
// "omp parallel for"), and the compiler takes it for granted, without any
// analysis. It is the most common construct in practice.
void saxpy_parallel(int n, float a, const float* x, float* y) {
    #pragma acc parallel loop copyin(x[0:n]) copy(y[0:n])
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}


// ---------------------------------------------------------------
// PART 2 : a reduction -> dot product
// ---------------------------------------------------------------
// Same reduction clause as in OpenMP : each gang / thread accumulates a
// private partial sum, combined at the end. In OpenACC 2.7 a reduction
// variable of a combined construct ("parallel loop") is copied back
// automatically ; copy(sum) makes it explicit for older compilers.
double dot(int n, const float* x, const float* y) {
    double sum = 0.0;
    #pragma acc parallel loop reduction(+:sum) copy(sum) copyin(x[0:n], y[0:n])
    for (int i = 0; i < n; ++i) {
        sum += static_cast<double>(x[i]) * y[i];
    }
    return sum;
}


int main() {

    // ---- where does the code run ? ----
#ifdef _OPENACC
    // _OPENACC = release date of the supported specification (yyyymm)
    std::cout << "compiled with OpenACC, _OPENACC = " << _OPENACC << std::endl;
    std::cout << "NVIDIA GPUs available : " << acc_get_num_devices(acc_device_nvidia)
              << std::endl;
    int on_host = 1;
    #pragma acc parallel copyout(on_host)
    {
        on_host = acc_on_device(acc_device_host);
    }
    std::cout << "compute regions run on : "
              << (on_host ? "the HOST (no GPU, or compiled for the CPU)" : "the GPU")
              << std::endl;
#else
    std::cout << "compiled WITHOUT OpenACC : everything runs sequentially" << std::endl;
#endif

    const int n = 1 << 24;  // 16 M floats = 64 MB per array
    float* x = new float[n];
    float* y = new float[n];

    // ---- part 1 ----
    std::cout << std::endl << "=== Part 1 : saxpy (y = 2 x + y, repeated 3 times) ==="
              << std::endl;

    auto run = [&](const char* name, void (*saxpy)(int, float, const float*, float*)) {
        for (int i = 0; i < n; ++i) {
            x[i] = 1.0f;
            y[i] = 1.0f;
        }
        double t0 = now();
        for (int r = 0; r < 3; ++r) saxpy(n, 2.0f, x, y);  // y = 1 + 3 * 2 = 7
        double t = now() - t0;
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (y[i] != 7.0f) { ok = false; break; }
        }
        std::cout << name << (ok ? "correct" : "WRONG") << " | " << t
                  << " s (transfers included)" << std::endl;
    };

    run("kernels                    : ", saxpy_kernels);
    run("kernels + loop independent : ", saxpy_kernels_independent);
    run("parallel loop              : ", saxpy_parallel);

    // ---- part 2 ----
    std::cout << std::endl << "=== Part 2 : dot product (reduction) ===" << std::endl;
    double t0 = now();
    double d = dot(n, x, y);  // 1 * 7 summed n times
    double t = now() - t0;
    std::cout << "dot = " << d << " (expected " << 7.0 * n << ") | " << t << " s" << std::endl;

    // free the allocated memory (no leak)
    delete[] x;
    delete[] y;
    return 0;
}
