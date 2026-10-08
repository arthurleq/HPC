#include <iostream>
#include <iomanip>
#include <vector>

#include <mpi.h>

// mpic++ -O2 6_MPI_communicators_topologies.cpp -o 6_MPI_communicators_topologies
// mpirun -np 6 --oversubscribe ./6_MPI_communicators_topologies
// (--oversubscribe : allows more processes than cores, fine for this demo)

// A COMMUNICATOR is a group of processes plus a private "context" : the
// messages of one communicator can never be mixed up with the messages of
// another one. MPI_COMM_WORLD contains every process, but new
// communicators can be created on subsets of processes, for instance to
// run a collective on a single row of a grid of processes. (Libraries
// also create their own copy with MPI_Comm_dup, so that their internal
// messages can't be confused with the user's ones.)
// A TOPOLOGY attaches a "shape" to a communicator : a 2D or 3D grid where
// each process knows its coordinates and its neighbours. It is the natural
// way to describe a domain decomposition (example 7).


// print a table gathered from every process, on rank 0, in rank order
// (values equal to -1 are printed as "-")
void print_table(const char* header, const std::vector<int>& mine, int rank, int size,
                 MPI_Comm comm) {
    int n = static_cast<int>(mine.size());
    std::vector<int> all(rank == 0 ? n * size : 0);
    MPI_Gather(mine.data(), n, MPI_INT, all.data(), n, MPI_INT, 0, comm);
    if (rank == 0) {
        std::cout << header << std::endl;
        for (int r = 0; r < size; ++r) {
            for (int k = 0; k < n; ++k) {
                int v = all[r * n + k];
                if (v == -1) std::cout << std::setw(8) << "-";
                else std::cout << std::setw(8) << v;
            }
            std::cout << std::endl;
        }
    }
}


// ---------------------------------------------------------------
// PART 1 : MPI_Comm_split -> rows and columns of a grid of processes
// ---------------------------------------------------------------
// MPI_Comm_split(comm, color, key, &newcomm) : the processes that give
// the SAME color end up in the same new communicator, ranked by 'key'.
//
// The processes are arranged in a grid, rank = row * ncols + col :
//
//        col 0   col 1
//       +-------+-------+
// row 0 |   0   |   1   |    row communicator    : color = row
//       +-------+-------+    column communicator : color = col
// row 1 |   2   |   3   |
//       +-------+-------+
// row 2 |   4   |   5   |
//       +-------+-------+
void part1_split(int rank, int size) {

    // MPI_Dims_create chooses a grid shape as square as possible
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int ncols = dims[1];
    int row = rank / ncols;
    int col = rank % ncols;

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);  // one communicator per row
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);  // one communicator per column

    int rank_in_row, rank_in_col;
    MPI_Comm_rank(row_comm, &rank_in_row);
    MPI_Comm_rank(col_comm, &rank_in_col);

    // a collective on row_comm only involves the processes of my row :
    // here, the sum of the world ranks of my row (and of my column)
    int row_sum, col_sum;
    MPI_Allreduce(&rank, &row_sum, 1, MPI_INT, MPI_SUM, row_comm);
    MPI_Allreduce(&rank, &col_sum, 1, MPI_INT, MPI_SUM, col_comm);

    if (rank == 0) {
        std::cout << "grid of " << dims[0] << " rows x " << dims[1] << " columns" << std::endl;
    }
    print_table("   world     row     col  rank in  rank in  sum of   sum of\n"
                "    rank                  row comm col comm  my row  my column",
                {rank, row, col, rank_in_row, rank_in_col, row_sum, col_sum},
                rank, size, MPI_COMM_WORLD);

    // communicators created by the program must be freed
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
}


// ---------------------------------------------------------------
// PART 2 : a Cartesian topology
// ---------------------------------------------------------------
// MPI_Cart_create builds a communicator with a grid shape, and then :
//   MPI_Cart_coords : rank -> coordinates in the grid
//   MPI_Cart_rank   : coordinates -> rank
//   MPI_Cart_shift  : ranks of my neighbours in one direction
// A dimension can be PERIODIC (the last process is the neighbour of the
// first one, like a ring) or not. At a non-periodic border, the missing
// neighbour is MPI_PROC_NULL : a send to (or a receive from) MPI_PROC_NULL
// does nothing and returns at once, so the code needs NO special case for
// the processes at the border of the domain.
//
// Here : dimension 0 (north/south) is NOT periodic, dimension 1
// (west/east) is periodic -> the grid is wrapped around a cylinder.
void part2_cartesian(int size) {

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {0, 1};  // not periodic north/south, periodic west/east
    int reorder = 0;          // 1 would let MPI renumber the processes to
                              // match the network topology of the machine
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart);

    int cart_rank;
    MPI_Comm_rank(cart, &cart_rank);
    int coords[2];
    MPI_Cart_coords(cart, cart_rank, 2, coords);

    // neighbours at distance 1 in each dimension
    int north, south, west, east;
    MPI_Cart_shift(cart, 0, 1, &north, &south);  // (row - 1, row + 1)
    MPI_Cart_shift(cart, 1, 1, &west, &east);    // (col - 1, col + 1)

    auto shown = [](int r) { return r == MPI_PROC_NULL ? -1 : r; };
    print_table("    rank     row     col   north   south    west    east",
                {cart_rank, coords[0], coords[1], shown(north), shown(south),
                 shown(west), shown(east)},
                cart_rank, size, cart);

    // MPI_Cart_sub keeps some dimensions and drops the others : keeping
    // only dimension 1 gives one communicator per row, like part 1
    int keep[2] = {0, 1};
    MPI_Comm row_comm;
    MPI_Cart_sub(cart, keep, &row_comm);
    int row_size;
    MPI_Comm_size(row_comm, &row_size);
    if (cart_rank == 0) {
        std::cout << "MPI_Cart_sub({0, 1}) : one communicator per row, of " << row_size
                  << " processes each" << std::endl;
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&cart);
}


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "=== Part 1 : MPI_Comm_split ===" << std::endl;
    part1_split(rank, size);

    if (rank == 0) std::cout << std::endl << "=== Part 2 : Cartesian topology ===" << std::endl;
    part2_cartesian(size);

    MPI_Finalize();
    return 0;
}
