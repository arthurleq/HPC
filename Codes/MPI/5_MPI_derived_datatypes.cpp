#include <iostream>
#include <iomanip>
#include <cstddef>  // offsetof

#include <mpi.h>

// mpic++ -O2 5_MPI_derived_datatypes.cpp -o 5_MPI_derived_datatypes
// mpirun -np 2 ./5_MPI_derived_datatypes
 
// MPI must know the memory layout of the data it sends. So far : arrays
// of a basic type (MPI_INT, MPI_DOUBLE...) stored contiguously.
// DERIVED datatypes describe more complex layouts, so that non-contiguous
// data or structures can be sent in ONE message, without copying them by
// hand into a temporary buffer first ("packing").
// A derived datatype is :
//   1. built with a constructor (MPI_Type_vector, MPI_Type_create_struct...)
//   2. committed with MPI_Type_commit before being used
//   3. freed with MPI_Type_free when it is not needed anymore

const int rows = 4;
const int cols = 6;

void print_matrix(const double m[rows][cols]) {
    for (int i = 0; i < rows; ++i) {
        std::cout << "    ";
        for (int j = 0; j < cols; ++j) std::cout << std::setw(5) << m[i][j];
        std::cout << std::endl;
    }
}


// ---------------------------------------------------------------
// PART 1 : a column of a matrix -> MPI_Type_vector
// ---------------------------------------------------------------
// In C/C++, a matrix is stored row by row (row-major) : a ROW is
// contiguous in memory, but a COLUMN is made of 'rows' elements, each one
// 'cols' elements after the previous one :
//
//     [  0  1  2  3  4  5 ]
//     [  6  7  8  9 10 11 ]     column 2 = elements 2, 8, 14, 20 :
//     [ 12 13 14 15 16 17 ]     'rows' blocks of 1 element,
//     [ 18 19 20 21 22 23 ]     with a stride of 'cols' elements
//
// MPI_Type_vector(count, blocklength, stride, oldtype, &newtype)
void part1_column(int rank) {

    MPI_Datatype column;
    MPI_Type_vector(rows, 1, cols, MPI_DOUBLE, &column);
    MPI_Type_commit(&column);

    double matrix[rows][cols];

    if (rank == 0) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) matrix[i][j] = i * cols + j;

        // ONE message, starting at the first element of column 2
        MPI_Send(&matrix[0][2], 1, column, 1, 0, MPI_COMM_WORLD);
        // the same column again, for the second receive below
        MPI_Send(&matrix[0][2], 1, column, 1, 1, MPI_COMM_WORLD);
    }
    else if (rank == 1) {
        // (a) into a CONTIGUOUS array : the receive type can be different
        // from the send type, only the sequence of basic types must match
        // (here : 4 doubles on both sides)
        double received[rows];
        MPI_Recv(received, rows, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "column 2 received into a contiguous array :";
        for (double v : received) std::cout << " " << v;
        std::cout << std::endl;

        // (b) directly into column 5 of a matrix, with the vector type
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) matrix[i][j] = 0.0;
        MPI_Recv(&matrix[0][5], 1, column, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "column 2 received into column 5 of a matrix of zeros :" << std::endl;
        print_matrix(matrix);
    }

    MPI_Type_free(&column);
}


// ---------------------------------------------------------------
// PART 2 : a block of a matrix -> MPI_Type_create_subarray
// ---------------------------------------------------------------
// The sub-block of 2 x 3 elements starting at row 1, column 2 :
//
//     [  0   1   2   3   4   5 ]
//     [  6   7 ( 8   9  10) 11 ]
//     [ 12  13 (14  15  16) 17 ]
//     [ 18  19  20  21  22  23 ]
//
// This is exactly what a process sends to its neighbours in a domain
// decomposition (see example 7).
void part2_subarray(int rank) {

    int sizes[2] = {rows, cols};    // shape of the whole array
    int subsizes[2] = {2, 3};       // shape of the block
    int starts[2] = {1, 2};         // position of its first element

    MPI_Datatype block;
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &block);
    MPI_Type_commit(&block);

    if (rank == 0) {
        double matrix[rows][cols];
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) matrix[i][j] = i * cols + j;

        // with a subarray type, the buffer is the START of the whole array
        MPI_Send(&matrix[0][0], 1, block, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1) {
        double received[2][3];
        MPI_Recv(&received[0][0], 6, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "sub-block received :" << std::endl;
        for (int i = 0; i < 2; ++i) {
            std::cout << "    ";
            for (int j = 0; j < 3; ++j) std::cout << std::setw(5) << received[i][j];
            std::cout << std::endl;
        }
    }

    MPI_Type_free(&block);
}


// ---------------------------------------------------------------
// PART 3 : a C++ structure -> MPI_Type_create_struct
// ---------------------------------------------------------------
// A structure mixes several types, and the compiler may insert PADDING
// between the fields to align them : here 4 bytes after 'id', so that
// 'position' starts on a multiple of 8 bytes. offsetof() gives the real
// position of each field, and the type must then be "resized" to
// sizeof(Particle), so that MPI finds the next particle of an array at
// the right place.
//
//   | id | pad |  position[0]  position[1]  position[2]  |  mass  |
//   0    4     8                                         32       40 bytes
struct Particle {
    int id;
    double position[3];
    double mass;
};

MPI_Datatype make_particle_type() {
    int block_lengths[3] = {1, 3, 1};
    MPI_Aint displacements[3] = {offsetof(Particle, id),
                                 offsetof(Particle, position),
                                 offsetof(Particle, mass)};
    MPI_Datatype types[3] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE};

    MPI_Datatype tmp, particle_type;
    MPI_Type_create_struct(3, block_lengths, displacements, types, &tmp);
    MPI_Type_create_resized(tmp, 0, sizeof(Particle), &particle_type);
    MPI_Type_commit(&particle_type);
    MPI_Type_free(&tmp);
    return particle_type;
}

void part3_struct(int rank) {

    MPI_Datatype particle_type = make_particle_type();

    if (rank == 0) {
        Particle particles[3] = {{7, {0.0, 1.0, 2.0}, 1.5},
                                 {8, {3.0, 4.0, 5.0}, 2.5},
                                 {9, {6.0, 7.0, 8.0}, 3.5}};
        MPI_Send(particles, 3, particle_type, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1) {
        // size   = number of bytes of data actually transferred
        // extent = distance between two consecutive particles in memory
        int size;
        MPI_Aint lower_bound, extent;
        MPI_Type_size(particle_type, &size);
        MPI_Type_get_extent(particle_type, &lower_bound, &extent);
        std::cout << "particle type : size = " << size << " bytes of data, extent = "
                  << extent << " bytes (= sizeof(Particle), padding included)" << std::endl;

        Particle particles[3];
        MPI_Recv(particles, 3, particle_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (const Particle& p : particles) {
            std::cout << "    particle " << p.id << " at (" << p.position[0] << ", "
                      << p.position[1] << ", " << p.position[2] << "), mass "
                      << p.mass << std::endl;
        }
    }

    MPI_Type_free(&particle_type);
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
    // only ranks 0 and 1 take part, the others have nothing to do. Rank 1
    // prints everything, so that the lines appear in the right order.

    if (rank == 1) std::cout << "=== Part 1 : a column (MPI_Type_vector) ===" << std::endl;
    part1_column(rank);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) std::cout << std::endl << "=== Part 2 : a block (MPI_Type_create_subarray) ==="
                             << std::endl;
    part2_subarray(rank);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) std::cout << std::endl << "=== Part 3 : a structure (MPI_Type_create_struct) ==="
                             << std::endl;
    part3_struct(rank);

    MPI_Finalize();
    return 0;
}
