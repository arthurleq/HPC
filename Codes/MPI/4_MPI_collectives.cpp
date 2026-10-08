#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <vector>

#include <mpi.h>

// mpic++ -O2 4_MPI_collectives.cpp -o 4_MPI_collectives
// mpirun -np 4 ./4_MPI_collectives

// A COLLECTIVE operation involves ALL the processes of a communicator :
// every process must call the same collective, in the same order (if one
// process skips it, the others wait forever). Collectives are implemented
// with optimized algorithms (trees, rings...) : a broadcast to P processes
// takes about log2(P) steps instead of P-1 sends from the root. Always
// prefer a collective to a hand-written loop of sends and receives.


// ---------------------------------------------------------------
// PART 1 : what each collective does with the data
// ---------------------------------------------------------------

// helper : rank 0 collects a small array from every process (with
// MPI_Gather, shown below) and prints them in the order of the ranks
void print_all(const char* title, const std::vector<int>& local, int rank, int size) {
    int n = static_cast<int>(local.size());
    std::vector<int> all(rank == 0 ? n * size : 0);
    MPI_Gather(local.data(), n, MPI_INT, all.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << title << std::endl;
        for (int r = 0; r < size; ++r) {
            std::cout << "    rank " << r << " :";
            for (int k = 0; k < n; ++k) std::cout << std::setw(5) << all[r * n + k];
            std::cout << std::endl;
        }
    }
}

void part1_data_movement(int rank, int size) {
    const int root = 0;

    // MPI_Bcast : the root sends the SAME buffer to everybody
    //   before : root has {1, 2, 3, 4}, the others {0, 0, 0, 0}
    std::vector<int> buffer(4, 0);
    if (rank == root) buffer = {1, 2, 3, 4};
    MPI_Bcast(buffer.data(), 4, MPI_INT, root, MPI_COMM_WORLD);
    print_all("MPI_Bcast     : the root's buffer {1, 2, 3, 4} is copied everywhere",
              buffer, rank, size);

    // MPI_Scatter : the root DEALS OUT its buffer, one piece per process
    //   before : root has {0, 10, 20, ...} (one element per process)
    std::vector<int> deck(rank == root ? size : 0);
    for (int r = 0; r < static_cast<int>(deck.size()); ++r) deck[r] = 10 * r;
    std::vector<int> card(1);
    MPI_Scatter(deck.data(), 1, MPI_INT, card.data(), 1, MPI_INT, root, MPI_COMM_WORLD);
    print_all("MPI_Scatter   : the root's {0, 10, 20, ...} is dealt out", card, rank, size);

    // MPI_Gather : the opposite, the root COLLECTS one piece from each
    std::vector<int> mine = {rank * rank};
    std::vector<int> collected(rank == root ? size : 0);
    MPI_Gather(mine.data(), 1, MPI_INT, collected.data(), 1, MPI_INT, root, MPI_COMM_WORLD);
    if (rank == root) {
        std::cout << "MPI_Gather    : the root collects rank*rank from everybody" << std::endl
                  << "    rank 0 :";
        for (int v : collected) std::cout << std::setw(5) << v;
        std::cout << "    (the other ranks receive nothing)" << std::endl;
    }

    // MPI_Allgather : a gather whose result is given to EVERYBODY
    std::vector<int> everybody(size);
    MPI_Allgather(mine.data(), 1, MPI_INT, everybody.data(), 1, MPI_INT, MPI_COMM_WORLD);
    print_all("MPI_Allgather : everybody collects rank*rank from everybody",
              everybody, rank, size);

    // MPI_Reduce : combines one value per process with an operation
    // (MPI_SUM, MPI_PROD, MPI_MAX, MPI_MIN, MPI_LAND...), result on the root
    int value = rank + 1;
    int sum = 0;
    MPI_Reduce(&value, &sum, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
    if (rank == root) {
        std::cout << "MPI_Reduce    : sum of (rank + 1) over all the ranks = " << sum
                  << " (result on the root only)" << std::endl;
    }

    // MPI_Allreduce : a reduction whose result is given to EVERYBODY
    int maximum = 0;
    MPI_Allreduce(&value, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    std::vector<int> shown = {maximum};
    print_all("MPI_Allreduce : max of (rank + 1), known by everybody", shown, rank, size);

    // MPI_Alltoall : everybody sends a DIFFERENT piece to each process :
    // piece j of rank i goes to rank j, at position i -> a transposition
    //   before : rank i has {10i + 0, 10i + 1, 10i + 2, ...}
    std::vector<int> send(size), recv(size);
    for (int j = 0; j < size; ++j) send[j] = 10 * rank + j;
    print_all("MPI_Alltoall  : before, rank i has 10*i + j at position j", send, rank, size);
    MPI_Alltoall(send.data(), 1, MPI_INT, recv.data(), 1, MPI_INT, MPI_COMM_WORLD);
    print_all("                after, rank j has 10*i + j at position i", recv, rank, size);
}


// ---------------------------------------------------------------
// PART 2 : computing pi -> Bcast + Reduce
// ---------------------------------------------------------------
// pi = integral from 0 to 1 of 4 / (1 + x^2) dx, approximated with the
// midpoint rule on n rectangles. The rectangles are dealt out cyclically :
// rank r takes rectangles r, r + P, r + 2P...
double compute_pi(int rank, int size) {
    long n = 0;
    if (rank == 0) n = 400000000;  // e.g. read from an input file by rank 0 only

    // everybody needs n
    MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    const double h = 1.0 / n;
    double local_sum = 0.0;
    for (long i = rank; i < n; i += size) {
        double x = (i + 0.5) * h;
        local_sum += 4.0 / (1.0 + x * x);
    }
    double local_pi = local_sum * h;

    // the partial results are added up on rank 0
    double pi = 0.0;
    MPI_Reduce(&local_pi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return pi;  // only meaningful on rank 0
}


// ---------------------------------------------------------------
// PART 3 : distributed statistics -> Scatterv, Allreduce, Gatherv, MAXLOC
// ---------------------------------------------------------------
// Rank 0 owns a data set of N values. It distributes it, everybody
// computes on its part, the data is normalized (z = (x - mean) / std) and
// collected back on rank 0.
// N is NOT a multiple of the number of processes : the "v" versions of
// the collectives (Scatterv, Gatherv) accept a different count per
// process, described by two arrays :
//   counts[r] = number of elements of rank r
//   displs[r] = position of the first one in the root's buffer
void part3_statistics(int rank, int size) {

    const int N = 1000003;

    // data set, only on rank 0
    std::vector<double> data;
    if (rank == 0) {
        data.resize(N);
        std::mt19937 gen(2024);
        std::normal_distribution<double> gauss(50.0, 10.0);
        for (double& x : data) x = gauss(gen);
    }

    // share N as evenly as possible : the first N % P ranks get one more
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = N / size + (r < N % size ? 1 : 0);
        displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
    }

    std::vector<double> local(counts[rank]);
    MPI_Scatterv(data.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 local.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // global mean : every process needs it for the next step -> Allreduce
    double local_sum = 0.0;
    for (double x : local) local_sum += x;
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double mean = global_sum / N;

    // global standard deviation
    double local_sq = 0.0;
    for (double x : local) local_sq += (x - mean) * (x - mean);
    double global_sq;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double stddev = std::sqrt(global_sq / N);

    // global maximum AND where it is : MPI_MAXLOC works on (value, index)
    // pairs, here the MPI_DOUBLE_INT type matching this struct
    struct { double value; int index; } local_max{-1e300, -1}, global_max;
    for (int k = 0; k < counts[rank]; ++k) {
        if (local[k] > local_max.value) {
            local_max.value = local[k];
            local_max.index = displs[rank] + k;  // index in the WHOLE data set
        }
    }
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

    // normalize the local part, then collect everything on rank 0
    for (double& x : local) x = (x - mean) / stddev;
    std::vector<double> normalized(rank == 0 ? N : 0);
    MPI_Gatherv(local.data(), counts[rank], MPI_DOUBLE,
                normalized.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "counts per rank :";
        for (int c : counts) std::cout << " " << c;
        std::cout << "   (N = " << N << ")" << std::endl;

        // sequential reference
        double s = 0.0, sq = 0.0;
        for (double x : data) s += x;
        double m = s / N;
        for (double x : data) sq += (x - m) * (x - m);
        double sd = std::sqrt(sq / N);
        double max_diff = 0.0;
        for (int i = 0; i < N; ++i) {
            max_diff = std::fmax(max_diff, std::fabs(normalized[i] - (data[i] - m) / sd));
        }

        std::cout << std::setprecision(10)
                  << "mean     : parallel " << mean << " | sequential " << m << std::endl
                  << "std dev  : parallel " << stddev << " | sequential " << sd << std::endl
                  << "maximum  : " << global_max.value << " at index " << global_max.index
                  << " (data[" << global_max.index << "] = " << data[global_max.index] << ")"
                  << std::endl
                  << "normalized data : max difference with the sequential version = "
                  << std::setprecision(3) << max_diff << std::endl;
    }
}


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "=== Part 1 : data movement of the collectives ===" << std::endl;
    part1_data_movement(rank, size);

    if (rank == 0) std::cout << std::endl << "=== Part 2 : pi with Bcast + Reduce ===" << std::endl;

    // MPI_Barrier : nobody goes on before everybody has arrived. Rarely
    // needed for correctness, it is used here to start the timing together.
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    double pi = compute_pi(rank, size);
    double t = MPI_Wtime() - t0;
    if (rank == 0) {
        std::cout << std::setprecision(15) << "pi ~ " << pi << " | error = "
                  << std::setprecision(3) << std::fabs(pi - M_PI) << " | time = " << t
                  << " s with " << size << " processes" << std::endl;
    }

    if (rank == 0) std::cout << std::endl << "=== Part 3 : distributed statistics ===" << std::endl;
    part3_statistics(rank, size);

    MPI_Finalize();
    return 0;
}
