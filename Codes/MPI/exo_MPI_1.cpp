#include <mpi.h>
#include <iostream>

// mpic++ exo_MPI_1.cpp -o exo_MPI_1
// mpirun -np 2 ./exo_MPI_1

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        int valeur = 42;
        // TODO: envoyer 'valeur' au processus 1
        MPI_Send(&valeur, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        
        
    } else if (rank == 1) {
        int valeur_recue;
        // TODO: recevoir la valeur depuis le processus 0
        MPI_Recv(&valeur_recue, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        std::cout << "Processus 1 a reçu : " << valeur_recue << std::endl;
    }

    MPI_Finalize();
    return 0;
}