#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include <mpi.h>

// mpic++ -O2 2_MPI_point_to_point.cpp -o 2_MPI_point_to_point
// mpirun -np 4 ./2_MPI_point_to_point
// mpirun -np 4 ./2_MPI_point_to_point deadlock     <- see part 3
 
// A message is made of :
//   - a BUFFER   : address, number of elements, MPI datatype (MPI_INT,
//                  MPI_DOUBLE, MPI_CHAR...)
//   - an ENVELOPE : source and destination ranks, a TAG (an integer chosen
//                  by the programmer to tell kinds of messages apart), and
//                  a communicator
//
//   MPI_Send(buffer, count, type, destination, tag, communicator)
//   MPI_Recv(buffer, count, type, source,      tag, communicator, &status)
//
// A receive matches a send when the source, the tag and the communicator
// match (MPI_ANY_SOURCE and MPI_ANY_TAG are wildcards). Both calls are
// BLOCKING : they return only when the buffer can be used again. MPI_Recv
// returns when the message has arrived ; MPI_Send returns when the data
// has been copied somewhere (sent, or stored in an internal buffer), which
// may or may not require the receiver to be ready (see part 3).


// ---------------------------------------------------------------
// PART 1 : send, receive and the status of a message
// ---------------------------------------------------------------
void part1_basics(int rank, int size) {

    // 1.a : rank 0 sends an array of 5 doubles to rank 1, with tag 42
    if (rank == 0) {
        double data[5] = {1.5, 2.5, 3.5, 4.5, 5.5};
        MPI_Send(data, 5, MPI_DOUBLE, 1, 42, MPI_COMM_WORLD);
    }
    else if (rank == 1) {
        double data[5];
        MPI_Status status;  // filled by MPI_Recv : who sent it, with which tag
        MPI_Recv(data, 5, MPI_DOUBLE, 0, 42, MPI_COMM_WORLD, &status);
        std::cout << "rank 1 received {" << data[0] << ", " << data[1] << ", ..., "
                  << data[4] << "} from rank " << status.MPI_SOURCE << " with tag "
                  << status.MPI_TAG << std::endl;
    }

    // the output of different processes is forwarded to the terminal by
    // mpirun : a barrier (example 4) doesn't fully guarantee the order of
    // the lines on the screen, but keeps the parts of this demo apart
    MPI_Barrier(MPI_COMM_WORLD);

    // 1.b : every other rank sends to rank 0 a message of a DIFFERENT
    // length (rank r sends 2r integers). Rank 0 knows neither who will send
    // first nor how many integers will come :
    //   MPI_Probe     waits for the next message (here from ANY source) and
    //                 describes it in 'status', without receiving it
    //   MPI_Get_count gives its number of elements
    // -> allocate a buffer of the right size, then receive THAT message
    if (rank != 0) {
        std::vector<int> message(2 * rank, rank);
        MPI_Send(message.data(), static_cast<int>(message.size()), MPI_INT, 0, 0,
                 MPI_COMM_WORLD);
    }
    else {
        for (int k = 1; k < size; ++k) {
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

            int count;
            MPI_Get_count(&status, MPI_INT, &count);

            std::vector<int> message(count);
            MPI_Recv(message.data(), count, MPI_INT, status.MPI_SOURCE, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "rank 0 received " << count << " integers from rank "
                      << status.MPI_SOURCE << " (in order of arrival)" << std::endl;
        }
    }
}


// ---------------------------------------------------------------
// PART 2 : ping-pong -> latency and bandwidth
// ---------------------------------------------------------------
// The time to send a message of n bytes is well described by
//     T(n) = latency + n / bandwidth
//   latency   : fixed cost of any message, even an empty one (software
//               layers, network hardware) ~ 0.2-1 us inside a node,
//               1-2 us between nodes on a fast network (InfiniBand)
//   bandwidth : bytes per second once the message is "flowing"
// Small messages are dominated by the latency, big ones by the bandwidth :
// it is better to send ONE big message than many small ones.
// Ping-pong : rank 0 sends n bytes to rank 1, which sends them back, many
// times ; the time of one message is half of the average round trip.
void part2_ping_pong(int rank) {

    if (rank == 0) {
        std::cout << "   message size |  time / message |   bandwidth" << std::endl;
    }

    for (int n = 8; n <= (16 << 20); n *= 8) {

        std::vector<char> buffer(n, 'x');
        const int repetitions = (n < 65536) ? 2000 : 100;

        // only ranks 0 and 1 play, the others wait at the barrier below
        if (rank <= 1) {
            double t0 = MPI_Wtime();  // wall-clock time in seconds
            for (int r = 0; r < repetitions; ++r) {
                if (rank == 0) {
                    MPI_Send(buffer.data(), n, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(buffer.data(), n, MPI_CHAR, 1, 0, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                }
                else {
                    MPI_Recv(buffer.data(), n, MPI_CHAR, 0, 0, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    MPI_Send(buffer.data(), n, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
                }
            }
            double t = (MPI_Wtime() - t0) / (2.0 * repetitions);

            if (rank == 0) {
                std::cout << std::setw(12) << n << " B | " << std::setw(12)
                          << std::fixed << std::setprecision(2) << t * 1e6
                          << " us | " << std::setw(7) << n / t / 1e9 << " GB/s"
                          << std::endl;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}


// ---------------------------------------------------------------
// PART 3 : exchanging in a ring -> the deadlock trap
// ---------------------------------------------------------------
// Each rank sends a message to its right neighbour and receives one from
// its left neighbour (the last rank's right neighbour is rank 0) :
//
//      rank 0 ---> rank 1 ---> rank 2 ---> rank 3
//        ^                                   |
//        +-----------------------------------+
//
// This "shift" is the basis of countless algorithms (systolic matrix
// products, halo exchanges...).

// UNSAFE : everybody calls MPI_Send first. A big message can't be stored
// in an internal buffer, so MPI_Send waits until the receiver has posted
// its MPI_Recv... but every rank is stuck in its MPI_Send, nobody ever
// reaches MPI_Recv : DEADLOCK, the program hangs forever.
// The vicious part : with SMALL messages, MPI copies the data into an
// internal buffer (the "eager" protocol) and MPI_Send returns at once, so
// the code works on small test cases and hangs on the real big ones.
void ring_unsafe(int right, int left, std::vector<double>& out, std::vector<double>& in) {
    int n = static_cast<int>(out.size());
    MPI_Send(out.data(), n, MPI_DOUBLE, right, 0, MPI_COMM_WORLD);
    MPI_Recv(in.data(), n, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// SAFE 1 : break the symmetry. Even ranks send then receive, odd ranks
// receive then send : every send always finds a matching receive.
void ring_ordered(int rank, int right, int left, std::vector<double>& out,
                  std::vector<double>& in) {
    int n = static_cast<int>(out.size());
    if (rank % 2 == 0) {
        MPI_Send(out.data(), n, MPI_DOUBLE, right, 0, MPI_COMM_WORLD);
        MPI_Recv(in.data(), n, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else {
        MPI_Recv(in.data(), n, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(out.data(), n, MPI_DOUBLE, right, 0, MPI_COMM_WORLD);
    }
}

// SAFE 2 (the simplest) : MPI_Sendrecv sends and receives at the same
// time, and MPI takes care of the ordering. Non-blocking communications
// (example 3) are the other classic solution.
void ring_sendrecv(int right, int left, std::vector<double>& out, std::vector<double>& in) {
    int n = static_cast<int>(out.size());
    MPI_Sendrecv(out.data(), n, MPI_DOUBLE, right, 0,
                 in.data(), n, MPI_DOUBLE, left, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// true if the received buffer contains what the left neighbour sent
// (every value equal to the left neighbour's rank) on EVERY process
bool check_ring(const std::vector<double>& in, int left) {
    int ok = 1;
    for (double v : in) {
        if (v != left) { ok = 0; break; }
    }
    // combine the flags of all the processes (collective operation, see
    // example 4) : the result is 1 only if every process has ok = 1
    int all_ok;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return all_ok == 1;
}


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) std::cerr << "this example needs at least 2 processes" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);  // stops ALL the processes
    }

    if (rank == 0) std::cout << "=== Part 1 : send / receive / status ===" << std::endl;
    part1_basics(rank, size);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) std::cout << std::endl << "=== Part 2 : ping-pong between ranks 0 and 1 ==="
                             << std::endl;
    part2_ping_pong(rank);

    if (rank == 0) std::cout << std::endl << "=== Part 3 : ring exchange ===" << std::endl;

    int right = (rank + 1) % size;
    int left = (rank - 1 + size) % size;
    // 8 MB per message : far too big to be buffered by MPI
    std::vector<double> out(1 << 20, static_cast<double>(rank));
    std::vector<double> in(1 << 20, -1.0);

    if (argc > 1 && std::string(argv[1]) == "deadlock") {
        if (rank == 0) {
            std::cout << "unsafe version : every rank calls MPI_Send first..." << std::endl
                      << "-> the program now hangs forever : press Ctrl+C to stop it"
                      << std::endl;
        }
        ring_unsafe(right, left, out, in);
        if (rank == 0) std::cout << "never printed" << std::endl;
    }

    ring_ordered(rank, right, left, out, in);
    bool ok = check_ring(in, left);
    if (rank == 0) {
        std::cout << "even/odd ordering : " << (ok ? "correct" : "WRONG") << std::endl;
    }

    std::fill(in.begin(), in.end(), -1.0);
    ring_sendrecv(right, left, out, in);
    ok = check_ring(in, left);
    if (rank == 0) {
        std::cout << "MPI_Sendrecv      : " << (ok ? "correct" : "WRONG") << std::endl;
    }

    MPI_Finalize();
    return 0;
}
