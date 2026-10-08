#include <iostream>
#include <iomanip>
#include <vector>

#include <mpi.h>

// mpic++ -O2 3_MPI_non_blocking.cpp -o 3_MPI_non_blocking
// mpirun -np 4 ./3_MPI_non_blocking

// Non-blocking communications : MPI_Isend and MPI_Irecv START a
// communication and return IMMEDIATELY, with a "request" that represents
// the communication in progress. Until the request is completed, the
// buffer must not be touched (no writing into a buffer being sent, no
// reading of a buffer being received). A request is completed with :
//   MPI_Wait(&request, &status)          blocks until it is done
//   MPI_Test(&request, &flag, &status)   checks, without blocking (flag = 1 if done)
//   MPI_Waitall / MPI_Waitany / MPI_Testall...   the same for several requests
//
// Two benefits :
//   1. no deadlock, whatever the order of the calls (part 1)
//   2. a process can work while its messages are in flight : this is the
//      overlapping of communications and computations (part 2)


// ---------------------------------------------------------------
// PART 1 : a ring exchange that can't deadlock
// ---------------------------------------------------------------
// The same shift as in example 2 (send to the right neighbour, receive
// from the left one), with big messages, but no ordering problem anymore :
// both communications are started, then we wait for both.
void ring_nonblocking(int right, int left, std::vector<double>& out,
                      std::vector<double>& in) {
    int n = static_cast<int>(out.size());
    MPI_Request requests[2];

    // posting the receive first is a good habit : the incoming message can
    // then be written directly into 'in' instead of a temporary buffer
    MPI_Irecv(in.data(), n, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(out.data(), n, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &requests[1]);

    // ... any computation that doesn't touch 'in' or 'out' could go here ...

    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
    // only now can 'in' be read and 'out' be modified
}


// ---------------------------------------------------------------
// PART 2 : hiding the waiting time behind computation
// ---------------------------------------------------------------
// Rank 1 is busy for 1 s before it can receive a message from rank 0.
// Rank 0 must send that message, and also has 1 s of computation to do,
// which doesn't depend on the message.
//
//   blocking     : rank 0 |--- waits for rank 1 ---|--- computes ---|      2 s
//   non-blocking : rank 0 |--- computes ---|wait|                         1 s
//                         (the message is delivered in the meantime)
//
// MPI_Ssend is the SYNCHRONOUS send : it completes only when the matching
// receive has started. Unlike MPI_Send, it never uses MPI's internal
// buffers, so its behaviour doesn't depend on the message size.

// simulates 'seconds' of computation (busy loop)
void compute_for(double seconds) {
    double t0 = MPI_Wtime();
    while (MPI_Wtime() - t0 < seconds) { }
}

double latency_hiding(int rank, bool non_blocking) {

    std::vector<double> message(1000, 3.14);
    double t0 = MPI_Wtime();

    if (rank == 0) {
        if (non_blocking) {
            MPI_Request request;
            MPI_Issend(message.data(), 1000, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &request);
            compute_for(1.0);                        // useful work while the message waits
            MPI_Wait(&request, MPI_STATUS_IGNORE);   // usually returns at once
        }
        else {
            MPI_Ssend(message.data(), 1000, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);  // waits ~1 s
            compute_for(1.0);
        }
    }
    else if (rank == 1) {
        compute_for(1.0);  // busy : can't receive before 1 s
        MPI_Recv(message.data(), 1000, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return MPI_Wtime() - t0;
}


// ---------------------------------------------------------------
// PART 3 : processing results in order of arrival -> MPI_Waitany
// ---------------------------------------------------------------
// A "manager / workers" pattern : every worker computes for a different
// time and sends its result to rank 0. Rank 0 posts one MPI_Irecv per
// worker, then MPI_Waitany returns as soon as ANY of them is complete :
// the results are processed in the order they arrive, not in the order
// of the ranks. (Here the last ranks finish first.)
void manager_workers(int rank, int size) {

    if (rank == 0) {
        std::vector<double> results(size);
        std::vector<MPI_Request> requests(size - 1);
        for (int w = 1; w < size; ++w) {
            MPI_Irecv(&results[w], 1, MPI_DOUBLE, w, 0, MPI_COMM_WORLD, &requests[w - 1]);
        }

        double t0 = MPI_Wtime();
        for (int k = 1; k < size; ++k) {
            int index;  // which request has completed
            MPI_Waitany(size - 1, requests.data(), &index, MPI_STATUS_IGNORE);
            int worker = index + 1;
            std::cout << "  after " << std::fixed << std::setprecision(2)
                      << MPI_Wtime() - t0 << " s : result of rank " << worker
                      << " = " << results[worker] << std::endl;
        }
    }
    else {
        compute_for(0.2 * (size - rank));  // rank size-1 is the fastest
        double result = 100.0 * rank;
        MPI_Send(&result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) std::cerr << "this example needs at least 2 processes" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ---- part 1 ----
    if (rank == 0) std::cout << "=== Part 1 : non-blocking ring exchange ===" << std::endl;

    int right = (rank + 1) % size;
    int left = (rank - 1 + size) % size;
    std::vector<double> out(1 << 20, static_cast<double>(rank));  // 8 MB
    std::vector<double> in(1 << 20, -1.0);

    ring_nonblocking(right, left, out, in);

    int ok = (in[0] == left && in.back() == left) ? 1 : 0;
    int all_ok;
    MPI_Reduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Isend / Irecv / Waitall : " << (all_ok ? "correct" : "WRONG")
                  << ", no deadlock" << std::endl;
    }

    // ---- part 2 ----
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << std::endl << "=== Part 2 : overlapping ===" << std::endl;

    double t_blocking = latency_hiding(rank, false);
    MPI_Barrier(MPI_COMM_WORLD);
    double t_non_blocking = latency_hiding(rank, true);
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(2)
                  << "blocking     (MPI_Ssend)  : rank 0 done after " << t_blocking
                  << " s" << std::endl
                  << "non-blocking (MPI_Issend) : rank 0 done after " << t_non_blocking
                  << " s" << std::endl;
    }

    // ---- part 3 ----
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << std::endl << "=== Part 3 : MPI_Waitany ===" << std::endl;
    manager_workers(rank, size);

    MPI_Finalize();
    return 0;
}
