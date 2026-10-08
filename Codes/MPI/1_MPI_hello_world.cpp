#include <iostream>
#include <string>
#include <unistd.h>  // getpid()

#include <mpi.h>

// mpic++ -O2 1_MPI_hello_world.cpp -o 1_MPI_hello_world
// mpirun -np 4 ./1_MPI_hello_world 

// MPI (Message Passing Interface) is the standard of DISTRIBUTED memory
// programming. mpirun launches N copies of the SAME program : N separate
// PROCESSES, possibly on different machines (nodes) of a cluster. Each
// process has its OWN memory : no variable is shared, the processes can
// only cooperate by explicitly sending each other messages.
//
//    mpirun -np 4 ./1_MPI_hello_world
//         |
//         +--> process of rank 0  (its own memory)
//         +--> process of rank 1  (its own memory)     they communicate
//         +--> process of rank 2  (its own memory)     by messages only
//         +--> process of rank 3  (its own memory)
//
// This model is called SPMD (Single Program, Multiple Data) : the code is
// the same for everybody, and each process uses its number, its "rank",
// to decide what to do and on which part of the data.


int main(int argc, char** argv) {

    // MPI_Init must be called before any other MPI function
    MPI_Init(&argc, &argv);

    // a COMMUNICATOR is a group of processes that can talk to each other.
    // MPI_COMM_WORLD contains all the processes launched by mpirun.
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // who am I ?        0 .. size-1
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // how many are we ?

    // name of the machine (node) this process runs on
    char node_name[MPI_MAX_PROCESSOR_NAME];
    int name_length;
    MPI_Get_processor_name(node_name, &name_length);

    // the processes are really separate programs : each one has its own
    // process id, given by the operating system
    std::cout << "hello from rank " << rank << " / " << size
              << " (pid " << getpid() << ") on node " << node_name << std::endl;

    // SPMD : same code for everybody, different behaviour depending on
    // the rank. By convention, rank 0 does the "administrative" work :
    // reading the input, printing the results...
    if (rank == 0) {
        int version, subversion;
        MPI_Get_version(&version, &subversion);

        char library[MPI_MAX_LIBRARY_VERSION_STRING];
        int length;
        MPI_Get_library_version(library, &length);
        std::string library_name(library, length);

        std::cout << "rank 0 : MPI standard " << version << "." << subversion
                  << ", implementation : "
                  << library_name.substr(0, library_name.find_first_of(",\n"))
                  << std::endl;
    }

    // every process owns its own copy of every variable : 'value' is a
    // DIFFERENT variable in each process, even if the code is the same.
    // Exchanging such values requires messages (next examples).
    double value = 10.0 * rank;
    std::cout << "rank " << rank << " : my value is " << value << std::endl;

    // MPI_Finalize must be the last MPI call of every process
    MPI_Finalize();
    return 0;
}
