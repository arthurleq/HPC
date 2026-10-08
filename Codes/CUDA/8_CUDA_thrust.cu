#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

// nvcc -O3 -arch=native 8_CUDA_thrust.cu -o 8_CUDA_thrust
// ./8_CUDA_thrust 

// THRUST is to CUDA what the STL is to C++ : containers and algorithms
// (transform, reduce, sort, scan, count...) that run on the GPU, without
// writing a single kernel. It comes with the CUDA toolkit, header-only.
//   thrust::device_vector<T> : an array in the GPU memory, allocated and
//                              freed automatically (RAII), copied with "="
//   thrust::host_vector<T>   : its counterpart in the host memory
// The algorithms take iterators, like the STL, and run on the device when
// given device iterators. The operations applied to the elements are
// FUNCTORS : structures whose operator() is __host__ __device__, so that
// the GPU can call it.
//
// Example 6 took 4 hand-written kernels to sum an array : here, 1 line.
// Thrust reports CUDA errors by throwing exceptions (thrust::system_error).

double now() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// a pseudo-random float in [0, 1) computed from an integer : the
// "finalizer" of the MurmurHash3 hash function (cheap and deterministic)
struct random_value {
    __host__ __device__ float operator()(unsigned int i) const {
        unsigned int h = i;
        h ^= h >> 16;
        h *= 0x85ebca6bu;
        h ^= h >> 13;
        h *= 0xc2b2ae35u;
        h ^= h >> 16;
        return (h & 0xffffff) / 16777216.0f;  // 24 random bits -> [0, 1)
    }
};

struct square {
    __host__ __device__ double operator()(float x) const { return static_cast<double>(x) * x; }
};

struct greater_than {
    float threshold;
    __host__ __device__ bool operator()(float x) const { return x > threshold; }
};


int main() try {

    const int n = 1 << 24;  // 16 M values

    // ---- fill a device vector : transform the sequence 0, 1, 2... ----
    // counting_iterator generates the integers on the fly : no array needed
    thrust::device_vector<float> d(n);
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(n),
                      d.begin(), random_value());

    // ---- reductions ----
    // (accumulated in double : 16 M floats summed in float would lose digits)
    double sum = thrust::reduce(d.begin(), d.end(), 0.0, thrust::plus<double>());
    double sum_sq = thrust::transform_reduce(d.begin(), d.end(), square(), 0.0,
                                             thrust::plus<double>());
    double mean = sum / n;
    double stddev = std::sqrt(sum_sq / n - mean * mean);
    auto minmax = thrust::minmax_element(d.begin(), d.end());
    long above = thrust::count_if(d.begin(), d.end(), greater_than{0.9f});

    std::cout << "=== statistics of " << n << " uniform random values in [0, 1) ===" << std::endl
              << std::setprecision(5)
              << "mean     = " << mean << "   (expected 0.5)" << std::endl
              << "std dev  = " << stddev << "   (expected 1/sqrt(12) = 0.28868)" << std::endl
              << "min, max = " << std::setprecision(8) << *minmax.first << ", "
              << *minmax.second << std::setprecision(5) << std::endl
              << "> 0.9    : " << above << " values (expected ~" << n / 10 << ")" << std::endl;
    // (*minmax.first reads ONE element from the GPU memory : convenient,
    // but each such access is a tiny transfer : avoid it in loops)

    // ---- sorting : GPU vs CPU ----
    thrust::host_vector<float> h = d;  // device -> host copy, in one line

    double t0 = now();
    std::sort(h.begin(), h.end());
    double t_cpu = now() - t0;

    t0 = now();
    thrust::sort(d.begin(), d.end());
    cudaDeviceSynchronize();
    double t_gpu = now() - t0;

    thrust::host_vector<float> sorted_on_gpu = d;
    bool same = thrust::equal(h.begin(), h.end(), sorted_on_gpu.begin());

    std::cout << std::endl << "=== sorting " << n << " floats ===" << std::endl
              << std::fixed << std::setprecision(1)
              << "std::sort on the CPU    : " << std::setw(7) << t_cpu * 1e3 << " ms" << std::endl
              << "thrust::sort on the GPU : " << std::setw(7) << t_gpu * 1e3 << " ms   -> "
              << std::setprecision(0) << t_cpu / t_gpu << " times faster" << std::endl
              << "same result : " << (same ? "yes" : "NO") << std::endl;

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "error : " << e.what() << std::endl;
    return 1;
}
