#include <iostream>
#include <iomanip>
#include <vector>

#include <mpi.h>

// mpic++ -O2 8_MPI_one_sided.cpp -o 8_MPI_one_sided
// mpirun -np 4 ./8_MPI_one_sided

// ONE-SIDED communications (or RMA, Remote Memory Access) : a process
// exposes a part of its memory in a WINDOW, and the other processes can
// then read it (MPI_Get), write it (MPI_Put) or update it (MPI_Accumulate)
// directly, WITHOUT the target process taking part : no matching receive.
// It maps well on the RDMA capabilities of HPC networks (InfiniBand...).
// 
//     two-sided (Send/Recv)                one-sided (Put/Get)
//   origin          target              origin          target
//   Send  ------->  Recv                Put   ------->  [window]   (target does nothing)
//
// The price to pay : synchronization becomes explicit. RMA operations
// happen inside "epochs", opened and closed by :
//   - MPI_Win_fence          : ACTIVE target, collective (all the processes
//                              of the window call it, like a barrier)
//   - MPI_Win_lock / unlock  : PASSIVE target, only the origin process is
//                              involved, the target goes on with its work
// The data of a Put / Get is only guaranteed to be transferred (visible)
// once the epoch is closed.


// ---------------------------------------------------------------
// PART 1 : MPI_Put with fences -> writing into rank 0's memory
// ---------------------------------------------------------------
// Rank 0 exposes an array of 'size' integers ; every process writes its
// result directly at its own position. The same result as an MPI_Gather,
// but rank 0 never calls a receive function.
void part1_put(int rank, int size) {

    // MPI_Win_allocate : allocates the memory AND creates the window over it
    // (better than MPI_Win_create on existing memory : MPI can choose memory
    // that the network or the other processes of the node access directly).
    // Only rank 0 exposes memory, the others take part with 0 bytes.
    int* results = nullptr;
    MPI_Aint bytes = (rank == 0) ? size * sizeof(int) : 0;
    MPI_Win win;
    MPI_Win_allocate(bytes, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &results, &win);
    if (rank == 0) {
        for (int r = 0; r < size; ++r) results[r] = -1;
    }

    int my_result = 100 + rank * rank;

    MPI_Win_fence(0, win);  // opens the epoch
    // put 1 int into the window of rank 0, at displacement 'rank'
    // (counted in units of disp_unit = sizeof(int))
    MPI_Put(&my_result, 1, MPI_INT, 0, rank, 1, MPI_INT, win);
    MPI_Win_fence(0, win);  // closes it : every Put is now complete

    if (rank == 0) {
        std::cout << "window of rank 0 after the Puts :";
        for (int r = 0; r < size; ++r) std::cout << " " << results[r];
        std::cout << "   (100 + rank^2 written by each rank)" << std::endl;
    }

    MPI_Win_free(&win);  // also frees the memory allocated by MPI_Win_allocate
}


// ---------------------------------------------------------------
// PART 2 : MPI_Get -> reading the memory of a neighbour
// ---------------------------------------------------------------
void part2_get(int rank, int size) {

    double* exposed = nullptr;
    MPI_Win win;
    MPI_Win_allocate(sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD,
                     &exposed, &win);
    *exposed = 1000.0 + rank;

    int right = (rank + 1) % size;
    double read_value = -1.0;

    MPI_Win_fence(0, win);
    MPI_Get(&read_value, 1, MPI_DOUBLE, right, 0, 1, MPI_DOUBLE, win);
    // read_value can NOT be used here yet : the Get is only started
    MPI_Win_fence(0, win);

    // gather the values on rank 0 to print them in order
    std::vector<double> all(rank == 0 ? size : 0);
    MPI_Gather(&read_value, 1, MPI_DOUBLE, all.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            std::cout << "rank " << r << " read " << all[r] << " in the window of rank "
                      << (r + 1) % size << std::endl;
        }
    }

    MPI_Win_free(&win);
}


// ---------------------------------------------------------------
// PART 3 : atomic operations, passive target -> dynamic load balancing
// ---------------------------------------------------------------
// 40 tasks of very different costs must be shared among the processes.
// A static split (the first 10 tasks to rank 0, the next 10 to rank 1...)
// leaves the processes with the cheap tasks idle while the unlucky one
// finishes. Instead, a global task counter lives in the window of rank 0,
// and each process repeatedly ATOMICALLY increments it to get the number
// of its next task : "self-scheduling", the MPI equivalent of OpenMP's
// schedule(dynamic), with no process dedicated to handing out the work
// (rank 0 works too).
// MPI_Fetch_and_op(&value, &old, type, target, displacement, op, win) :
// atomically does  old = counter ; counter = counter op value.

const int n_tasks = 40;

// simulates the work of a task : busy for 'seconds'
void do_task(double seconds) {
    double t0 = MPI_Wtime();
    while (MPI_Wtime() - t0 < seconds) { }
}

// the tasks get more and more expensive : from 2 ms to 80 ms
double task_cost(int task) {
    return 0.002 * (task + 1);
}

// static split, for comparison : returns the time of the slowest process
double run_static(int rank, int size) {
    double t0 = MPI_Wtime();
    for (int task = rank * n_tasks / size; task < (rank + 1) * n_tasks / size; ++task) {
        do_task(task_cost(task));
    }
    double elapsed = MPI_Wtime() - t0, max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return max_elapsed;
}

void part3_dynamic(int rank, int size) {

    double t_static = run_static(rank, size);
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(3)
                  << "static split (blocks of consecutive tasks) : finished after "
                  << t_static << " s" << std::endl;
    }

    int* counter = nullptr;
    MPI_Aint bytes = (rank == 0) ? sizeof(int) : 0;
    MPI_Win win;
    MPI_Win_allocate(bytes, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &counter, &win);
    if (rank == 0) *counter = 0;
    MPI_Barrier(MPI_COMM_WORLD);  // the counter is initialized before anybody uses it

    int tasks_done = 0;
    double busy_time = 0.0;
    const int one = 1;
    double t0 = MPI_Wtime();

    while (true) {
        int task;
        // passive target epoch on rank 0 : rank 0 doesn't need to do anything
        MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
        MPI_Fetch_and_op(&one, &task, MPI_INT, 0, 0, MPI_SUM, win);
        MPI_Win_unlock(0, win);  // after unlock, 'task' holds the old value

        if (task >= n_tasks) break;  // no task left

        do_task(task_cost(task));
        busy_time += task_cost(task);
        ++tasks_done;
    }
    double elapsed = MPI_Wtime() - t0;

    // report : how many tasks each process did
    std::vector<int> done(rank == 0 ? size : 0);
    std::vector<double> busy(rank == 0 ? size : 0);
    MPI_Gather(&tasks_done, 1, MPI_INT, done.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&busy_time, 1, MPI_DOUBLE, busy.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double total = 0.0;
        for (int t = 0; t < n_tasks; ++t) total += task_cost(t);
        std::cout << "dynamic (shared task counter)            : finished after " << max_elapsed
                  << " s (ideal : total work / " << size << " = " << total / size << " s)"
                  << std::endl;
        for (int r = 0; r < size; ++r) {
            std::cout << "  rank " << r << " did " << std::setw(2) << done[r]
                      << " tasks, busy " << busy[r] << " s" << std::endl;
        }
    }

    MPI_Win_free(&win);
}


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "=== Part 1 : MPI_Put + MPI_Win_fence ===" << std::endl;
    part1_put(rank, size);

    if (rank == 0) std::cout << std::endl << "=== Part 2 : MPI_Get ===" << std::endl;
    part2_get(rank, size);

    if (rank == 0) std::cout << std::endl << "=== Part 3 : MPI_Fetch_and_op, dynamic load balancing ==="
                             << std::endl;
    part3_dynamic(rank, size);

    MPI_Finalize();
    return 0;
}
