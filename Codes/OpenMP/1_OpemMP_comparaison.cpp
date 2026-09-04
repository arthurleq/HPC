#include <iostream>

// initialisation of OpenMP environment variables
#include <omp.h>

// creation of a table of size n
void create_table(int n, int* table) {

    // parallel region to initialize the table
    #pragma omp parallel for

    for (int i = 0; i < n; ++i) {
        // initialize each element with its index
        table[i] = i; 
    }
}

// fuction for the sum of the table elements
// without any protection (not thread-safe)
long sum_table_unprotected(int n, int* table) {
    
    // variable to hold the sum (long type to avoid overflow for large n)
    long sum = 0;

    #pragma omp parallel for

    for (int i = 0; i < n; ++i) {
        sum += table[i]; // add each element to the sum
    }

    // return the computed sum
    return sum;
}


// fuction for the sum of the table elements
// with reduction
// Each thread cumulates in a private copy of sum (no shared memory during the loop), 
// and the final merge is done only once, at the end. Zero synchronization per iteration -> almost no cost.
long sum_table_reduction(int n, int* table) {
    
    // variable to hold the sum (long type to avoid overflow for large n)
    long sum = 0;

    #pragma omp parallel for reduction(+:sum)

    for (int i = 0; i < n; ++i) {
        sum += table[i]; // add each element to the sum
    }

    // return the computed sum
    return sum;
}

// fuction for the sum of the table elements
// with atomic
// Use an dedicated machine instruction (typically lock xadd on x86) 
// that guarantees atomicity at the memory/cache bus level for a simple operation like +=. 
// It's fast compared to a generic lock, but it still involves hardware synchronization on each iteration.
long sum_table_atomic(int n, int* table) {
    
    // variable to hold the sum (long type to avoid overflow for large n)
    long sum = 0;

    #pragma omp parallel for

    for (int i = 0; i < n; ++i) {
        #pragma omp atomic
        sum += table[i]; // add each element to the sum
    }

    // return the computed sum
    return sum;
}

// fuction for the sum of the table elements
// with critical
// Implement a generic critical section (lock/mutex), able to protect any arbitrarily complex block of code, 
// not just a scalar operation. This mechanism is much heavier.
long sum_table_critical(int n, int* table) {
    
    // variable to hold the sum (long type to avoid overflow for large n)
    long sum = 0;

    #pragma omp parallel for

    for (int i = 0; i < n; ++i) {
        #pragma omp critical
        sum += table[i]; // add each element to the sum
    }

    // return the computed sum
    return sum;
}


int main() {

    // size of the table
    const int n = static_cast<int>(10e6);

    // dynamically allocate memory for the table
    int* table = new int[n];

    // call the function to create and initialize the table
    create_table(n, table);

    // call the function to compute the sum of the table elements
    double t0 = omp_get_wtime();
    long total_sum = sum_table_unprotected(n, table);
    double t1 = omp_get_wtime();
    std::cout << "The sum unprotected of the table elements is: " << total_sum << std::endl;
    std::cout << "Time unprotected: " << (t1 - t0) << " s" << std::endl;

    // call the function to compute the sum of the table elements with reduction
    t0 = omp_get_wtime();
    long total_sum_reduction = sum_table_reduction(n, table);
    t1 = omp_get_wtime();
    std::cout << "The sum with reduction of the table elements is: " << total_sum_reduction << std::endl;
    std::cout << "Time reduction: " << (t1 - t0) << " s" << std::endl;

    // call the function to compute the sum of the table elements with atomic
    t0 = omp_get_wtime();
    long total_sum_atomic = sum_table_atomic(n, table);
    t1 = omp_get_wtime();
    std::cout << "The sum with atomic of the table elements is: " << total_sum_atomic << std::endl;
    std::cout << "Time atomic: " << (t1 - t0) << " s" << std::endl;

    // call the function to compute the sum of the table elements with critical
    t0 = omp_get_wtime();
    long total_sum_critical = sum_table_critical(n, table);
    t1 = omp_get_wtime();
    std::cout << "The sum with critical of the table elements is: " << total_sum_critical << std::endl;
    std::cout << "Time critical: " << (t1 - t0) << " s" << std::endl;

    // compute the expected sum using the formula for the sum of the first n-1 integers
    long reel_sum = static_cast<long>(n) * (n - 1) / 2;
    std::cout << "The expected sum is: " << reel_sum << std::endl;
    
    // free the allocated memory (no leak)
    delete[] table; 
    return 0;
}