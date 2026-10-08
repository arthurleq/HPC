#include <iostream>
#include <algorithm>
#include <climits>
#include <vector>

// nvc++ -acc -Minfo=accel -O2 4_OpenACC_reductions_atomics.cpp -o 4_OpenACC_reductions_atomics
// g++ -fopenacc -O2 4_OpenACC_reductions_atomics.cpp -o 4_OpenACC_reductions_atomics
 
// Thousands of GPU threads updating the same variable : the problem of
// example 1 of the OpenMP folder, with the same two answers :
//   - reduction(op:var) when the result is ONE value combined with an
//     associative operation : +, *, max, min, &, |, ^, &&, ||
//   - atomic when every thread must update a shared memory location that
//     depends on its data (a histogram bin, a counter...)
//     #pragma acc atomic update   x++ ; x += ... ; x = x * ...
//     #pragma acc atomic capture  v = x++ ; (read the old value AND update)
//     #pragma acc atomic read / write

// a pseudo-random 64-bit number computed from i : the "finalizer" of the
// MurmurHash3 hash function, cheap and deterministic. "acc routine seq"
// compiles the function for the device too (called by one thread at a time)
#pragma acc routine seq
inline unsigned long mix(unsigned long x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdul;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ul;
    x ^= x >> 33;
    return x;
}

// value number i of the data set : the average of two pseudo-random
// integers of [0, 1000), so that the middle values are the most frequent
// (a "triangular" distribution)
#pragma acc routine seq
inline int value_of(long i) {
    unsigned long h = mix(static_cast<unsigned long>(i));
    return static_cast<int>(((h & 0xffffffff) % 1000 + (h >> 32) % 1000) / 2);
}


int main() {

    const int rows = 2000, cols = 5000;
    const long n = static_cast<long>(rows) * cols;  // 10 M values

    int* v = new int[n];
    for (long i = 0; i < n; ++i) v[i] = value_of(i);

    // sequential references
    long ref_sum = 0;
    int ref_min = INT_MAX, ref_max = INT_MIN;
    long ref_high = 0;
    const int nbins = 10;
    long ref_hist[nbins] = {0};
    std::vector<long> ref_selected;
    for (long i = 0; i < n; ++i) {
        ref_sum += v[i];
        ref_min = std::min(ref_min, v[i]);
        ref_max = std::max(ref_max, v[i]);
        if (v[i] >= 500) ++ref_high;
        ref_hist[v[i] / 100]++;
        if (v[i] >= 995) ref_selected.push_back(i);
    }

    // the data set is copied once to the device for the 3 parts
    #pragma acc data copyin(v[0:n])
    {
        // ---------------------------------------------------------------
        // PART 1 : reductions
        // ---------------------------------------------------------------
        // several reductions in the same loop, each with its own operator
        long sum = 0;
        int vmin = INT_MAX, vmax = INT_MIN;

        #pragma acc parallel loop reduction(+:sum) reduction(min:vmin) reduction(max:vmax) \
                    copy(sum, vmin, vmax) present(v[0:n])
        for (long i = 0; i < n; ++i) {
            sum += v[i];
            vmin = (v[i] < vmin) ? v[i] : vmin;
            vmax = (v[i] > vmax) ? v[i] : vmax;
        }

        // a reduction over a 2D loop nest : collapse(2) merges the two loops,
        // and the reduction covers the whole merged iteration space
        long high = 0;
        #pragma acc parallel loop collapse(2) reduction(+:high) copy(high) present(v[0:n])
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (v[static_cast<long>(r) * cols + c] >= 500) high += 1;
            }
        }

        std::cout << "=== Part 1 : reductions ===" << std::endl
                  << "sum = " << sum << ", min = " << vmin << ", max = " << vmax
                  << ", values >= 500 : " << high << "   -> "
                  << ((sum == ref_sum && vmin == ref_min && vmax == ref_max && high == ref_high)
                          ? "correct" : "WRONG")
                  << std::endl;

        // ---------------------------------------------------------------
        // PART 2 : atomic update -> a histogram
        // ---------------------------------------------------------------
        // 10 M threads, 10 bins : many threads hit the same bin at the same
        // time. Without "atomic", increments are lost (race condition, as
        // with OpenMP). Atomics on the GPU memory are fast, but the updates
        // of ONE address are still serialized : the fewer the bins, the
        // higher the contention.
        long hist[nbins] = {0};

        #pragma acc parallel loop copy(hist[0:nbins]) present(v[0:n])
        for (long i = 0; i < n; ++i) {
            int bin = v[i] / 100;
            #pragma acc atomic update
            hist[bin] += 1;
        }

        bool ok = true;
        for (int b = 0; b < nbins; ++b) ok = ok && (hist[b] == ref_hist[b]);
        std::cout << std::endl << "=== Part 2 : histogram with atomic update ===" << std::endl;
        for (int b = 0; b < nbins; ++b) {
            std::cout << "  [" << 100 * b << ", " << 100 * (b + 1) << ") : " << hist[b]
                      << std::endl;
        }
        std::cout << "  -> " << (ok ? "correct" : "WRONG") << std::endl;

        // ---------------------------------------------------------------
        // PART 3 : atomic capture -> stream compaction
        // ---------------------------------------------------------------
        // Keep only the indices of the values >= 995 in a compact array.
        // Each thread that finds such a value reserves the next free slot :
        // "k = count++" must be done atomically, read AND increment, so that
        // two threads never get the same slot : that is "atomic capture".
        // The output array is allocated for the worst case (n), but only the
        // 'count' useful elements are brought back to the host.
        long* selected = new long[n];
        int count = 0;

        #pragma acc data create(selected[0:n])
        {
            #pragma acc parallel loop copy(count) present(v[0:n], selected[0:n])
            for (long i = 0; i < n; ++i) {
                if (v[i] >= 995) {
                    int k;
                    #pragma acc atomic capture
                    k = count++;
                    selected[k] = i;
                }
            }

            // only the first 'count' elements : a partial transfer
            #pragma acc update self(selected[0:count])
        }

        std::cout << std::endl << "=== Part 3 : compaction with atomic capture ===" << std::endl;
        std::cout << count << " values >= 995 (expected " << ref_selected.size()
                  << "), first indices found :";
        for (int k = 0; k < 6 && k < count; ++k) std::cout << " " << selected[k];
        std::cout << " ..." << std::endl;

        // on a GPU, the slots are taken in the order the threads arrive :
        // the SET of indices is right, but their ORDER changes from one run
        // to the other. Sorting them gives back the sequential result.
        std::sort(selected, selected + count);
        bool same = (count == static_cast<int>(ref_selected.size())) &&
                    std::equal(selected, selected + count, ref_selected.begin());
        std::cout << "after sorting, same indices as the sequential version : "
                  << (same ? "yes" : "NO") << std::endl;

        delete[] selected;
    }

    // free the allocated memory (no leak)
    delete[] v;
    return 0;
}
