#include <iostream>

// g++ -fopenacc exo_OpenACC_1.cpp -o exo_OpenACC_1


long sum_table_acc(int n, int* table) {
    long sum = 0;

    #pragma acc parallel loop reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += table[i];
    }
    return sum;
}

int main() {
    const int n = 1000;
    int* table = new int[n];
    for (int i = 0; i < n; ++i) table[i] = i;

    long total = sum_table_acc(n, table);
    std::cout << "Somme : " << total << std::endl;

    long expected = static_cast<long>(n) * (n - 1) / 2;
    std::cout << "Attendu : " << expected << std::endl;

    delete[] table;
    return 0;
}