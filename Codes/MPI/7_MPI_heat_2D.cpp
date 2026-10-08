#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <vector>

#include <mpi.h>

// mpic++ -O2 7_MPI_heat_2D.cpp -o 7_MPI_heat_2D
// mpirun -np 4 ./7_MPI_heat_2D

// Heat diffusion in a square plate : a hot disk (100 degrees) in the
// middle of a cold plate (0 degrees), whose borders are kept at 0. At each
// time step, every cell moves towards the average of its 4 neighbours
// (explicit finite differences for the heat equation dT/dt = D laplacian(T)) :
//
//   T_new[i][j] = T[i][j] + alpha * (T[i-1][j] + T[i+1][j] + T[i][j-1] + T[i][j+1] - 4 T[i][j])
//
// This is THE classic example of DOMAIN DECOMPOSITION, the way almost
// every simulation code (CFD, climate, mechanics...) runs on a cluster :
//
// 1. the grid is cut into blocks, one per process (2D Cartesian topology,
//    example 6) ; each process only stores and updates its own block
// 2. to update the cells at the edge of its block, a process needs the
//    neighbouring cells, which belong to its neighbours. Each block is
//    therefore surrounded by a layer of GHOST cells (or "halo"), copies of
//    the neighbours' edge cells, refreshed by messages at every time step :
//
//            +---------------------+
//            | g g g g g g g g g g |   g = ghost cells : received from the
//            | g o o o o o o o o g |       4 neighbours (north, south,
//            | g o o o o o o o o g |       west, east) before each step
//            | g o o o o o o o o g |
//            | g o o o o o o o o g |   o = cells owned and updated by
//            | g g g g g g g g g g |       this process
//            +---------------------+
//
// 3. global quantities (total heat, maximum temperature) are obtained
//    with reductions (example 4)
//
// The amount of computation of a block grows like its AREA, while the
// amount of communication grows like its PERIMETER : the bigger the
// blocks, the more efficient the parallelization (surface-to-volume ratio).

const int N = 512;              // the plate has N x N cells (+ a border of fixed cells)
const int steps = 5000;         // number of time steps
const int report_every = 1000;  // print global diagnostics every ... steps
const double alpha = 0.2;       // D dt / dx^2, must be <= 0.25 for stability


// one time step on a block of ny x nx cells, surrounded by one layer of
// ghost (or border) cells : a row is nx + 2 values long in memory.
// The SAME function is used by the parallel code (on each block) and by
// the sequential reference (on the whole plate).
void time_step(int ny, int nx, const double* t, double* t_new) {
    const int w = nx + 2;
    for (int i = 1; i <= ny; ++i) {
        for (int j = 1; j <= nx; ++j) {
            t_new[i * w + j] = t[i * w + j] + alpha * (t[(i - 1) * w + j] + t[(i + 1) * w + j]
                                                      + t[i * w + j - 1] + t[i * w + j + 1]
                                                      - 4.0 * t[i * w + j]);
        }
    }
}

// initial temperature of the cell (gi, gj) of the plate (global indices,
// 1 .. N for the cells of the plate, 0 and N+1 for the fixed border)
double initial_temperature(int gi, int gj) {
    double di = gi - (N + 1) / 2.0;
    double dj = gj - (N + 1) / 2.0;
    return (di * di + dj * dj < (N / 6.0) * (N / 6.0)) ? 100.0 : 0.0;
}

// the cells of a dimension of size n are shared among p blocks ; block
// number b gets 'count' cells starting at global index 'first' (1-based)
void block_range(int n, int p, int b, int& first, int& count) {
    count = n / p + (b < n % p ? 1 : 0);
    first = 1 + b * (n / p) + std::min(b, n % p);
}


// ---------------------------------------------------------------
// sequential reference (run by rank 0 at the end, to check the result)
// ---------------------------------------------------------------
std::vector<double> solve_sequential() {
    const int w = N + 2;  // with the border
    std::vector<double> t(w * w), t_new(w * w);
    for (int i = 0; i < w; ++i)
        for (int j = 0; j < w; ++j) {
            bool border = (i == 0 || j == 0 || i == N + 1 || j == N + 1);
            t[i * w + j] = t_new[i * w + j] = border ? 0.0 : initial_temperature(i, j);
        }

    for (int s = 0; s < steps; ++s) {
        time_step(N, N, t.data(), t_new.data());
        std::swap(t, t_new);
    }
    return t;
}


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- 1. a 2D grid of processes, not periodic ----
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart);

    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);
    int north, south, west, east;  // MPI_PROC_NULL at the border of the plate
    MPI_Cart_shift(cart, 0, 1, &north, &south);
    MPI_Cart_shift(cart, 1, 1, &west, &east);

    // ---- 2. my block : rows [i0, i0 + ny) and columns [j0, j0 + nx) ----
    int i0, ny, j0, nx;
    block_range(N, dims[0], coords[0], i0, ny);
    block_range(N, dims[1], coords[1], j0, nx);

    // local arrays : my ny x nx cells + 1 layer of ghost cells all around.
    // Local cell (i, j), with i in 0 .. ny+1 and j in 0 .. nx+1, is global
    // cell (i0 - 1 + i, j0 - 1 + j). The ghost cells that lie on the border
    // of the plate are never received (neighbour = MPI_PROC_NULL) and keep
    // their initial value 0 : they ARE the boundary condition.
    const int w = nx + 2;
    std::vector<double> t((ny + 2) * w, 0.0), t_new((ny + 2) * w, 0.0);
    for (int i = 1; i <= ny; ++i)
        for (int j = 1; j <= nx; ++j)
            t[i * w + j] = initial_temperature(i0 - 1 + i, j0 - 1 + j);

    if (rank == 0) {
        std::cout << "plate of " << N << " x " << N << " cells, " << steps << " time steps, "
                  << size << " processes as a " << dims[0] << " x " << dims[1]
                  << " grid (blocks of about " << ny << " x " << nx << " cells)" << std::endl;
    }

    // a column of my block (ny cells, one every w cells) : derived datatype
    // (example 5). A row is contiguous and needs no special type.
    MPI_Datatype column;
    MPI_Type_vector(ny, 1, w, MPI_DOUBLE, &column);
    MPI_Type_commit(&column);

    MPI_Barrier(cart);
    double t0 = MPI_Wtime();

    for (int s = 0; s < steps; ++s) {

        // ---- 3. halo exchange : 4 shifts, no deadlock thanks to Sendrecv.
        // At the border of the plate the neighbour is MPI_PROC_NULL and the
        // call does nothing : no special case in the code.
        //   my first row  -> north ; from south -> my bottom ghost row
        MPI_Sendrecv(&t[1 * w + 1], nx, MPI_DOUBLE, north, 0,
                     &t[(ny + 1) * w + 1], nx, MPI_DOUBLE, south, 0, cart, MPI_STATUS_IGNORE);
        //   my last row   -> south ; from north -> my top ghost row
        MPI_Sendrecv(&t[ny * w + 1], nx, MPI_DOUBLE, south, 1,
                     &t[0 * w + 1], nx, MPI_DOUBLE, north, 1, cart, MPI_STATUS_IGNORE);
        //   my first column -> west ; from east -> my right ghost column
        MPI_Sendrecv(&t[1 * w + 1], 1, column, west, 2,
                     &t[1 * w + nx + 1], 1, column, east, 2, cart, MPI_STATUS_IGNORE);
        //   my last column  -> east ; from west -> my left ghost column
        MPI_Sendrecv(&t[1 * w + nx], 1, column, east, 3,
                     &t[1 * w + 0], 1, column, west, 3, cart, MPI_STATUS_IGNORE);

        // ---- 4. update of my cells : exactly the sequential code ----
        time_step(ny, nx, t.data(), t_new.data());
        std::swap(t, t_new);

        // ---- 5. global diagnostics from time to time (a reduction is a
        // global synchronization : not at every step) ----
        if ((s + 1) % report_every == 0) {
            double local_heat = 0.0, local_max = 0.0;
            for (int i = 1; i <= ny; ++i)
                for (int j = 1; j <= nx; ++j) {
                    local_heat += t[i * w + j];
                    local_max = std::max(local_max, t[i * w + j]);
                }
            double heat, max_temperature;
            MPI_Reduce(&local_heat, &heat, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
            MPI_Reduce(&local_max, &max_temperature, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
            if (rank == 0) {
                std::cout << "  step " << std::setw(5) << s + 1 << " : total heat = "
                          << std::fixed << std::setprecision(0) << heat
                          << ", max temperature = " << std::setprecision(2)
                          << max_temperature << std::endl;
            }
        }
    }

    MPI_Barrier(cart);
    double t_parallel = MPI_Wtime() - t0;

    // ---- 6. gather the whole plate on rank 0 ----
    // Each process sends the INTERIOR of its local array (a subarray type
    // that skips the ghost cells). Rank 0 receives each block directly at
    // its place in the global array (another subarray type, one per block).
    MPI_Datatype interior;
    int local_sizes[2] = {ny + 2, nx + 2}, local_sub[2] = {ny, nx}, local_start[2] = {1, 1};
    MPI_Type_create_subarray(2, local_sizes, local_sub, local_start, MPI_ORDER_C, MPI_DOUBLE,
                             &interior);
    MPI_Type_commit(&interior);

    if (rank != 0) {
        MPI_Send(t.data(), 1, interior, 0, 0, cart);
    }
    else {
        std::vector<double> plate((N + 2) * (N + 2), 0.0);
        for (int r = 0; r < size; ++r) {
            int c[2];
            MPI_Cart_coords(cart, r, 2, c);
            int ri0, rny, rj0, rnx;
            block_range(N, dims[0], c[0], ri0, rny);
            block_range(N, dims[1], c[1], rj0, rnx);

            int sizes[2] = {N + 2, N + 2}, sub[2] = {rny, rnx}, start[2] = {ri0, rj0};
            MPI_Datatype block;
            MPI_Type_create_subarray(2, sizes, sub, start, MPI_ORDER_C, MPI_DOUBLE, &block);
            MPI_Type_commit(&block);
            if (r == 0) {
                // my own block : a "message to myself" with sendrecv
                MPI_Sendrecv(t.data(), 1, interior, 0, 0, plate.data(), 1, block, 0, 0,
                             MPI_COMM_SELF, MPI_STATUS_IGNORE);
            }
            else {
                MPI_Recv(plate.data(), 1, block, r, 0, cart, MPI_STATUS_IGNORE);
            }
            MPI_Type_free(&block);
        }

        // ---- 7. check against the sequential version ----
        double t1 = MPI_Wtime();
        std::vector<double> reference = solve_sequential();
        double t_sequential = MPI_Wtime() - t1;

        double max_diff = 0.0;
        for (int k = 0; k < (N + 2) * (N + 2); ++k) {
            max_diff = std::max(max_diff, std::fabs(plate[k] - reference[k]));
        }
        std::cout << std::setprecision(3)
                  << "parallel   : " << t_parallel << " s" << std::endl
                  << "sequential : " << t_sequential << " s   -> speedup "
                  << t_sequential / t_parallel << std::endl
                  << "max difference with the sequential result : " << max_diff << std::endl;

        // ---- 8. a coarse picture of the plate ----
        const char* shades = " .:-=+*#%@";
        std::cout << std::endl << "temperature map (' ' = 0 ... '@' = 100) :" << std::endl;
        for (int r = 0; r < 32; ++r) {
            std::cout << "  |";
            for (int c = 0; c < 64; ++c) {
                int gi = 1 + r * N / 32 + N / 64;
                int gj = 1 + c * N / 64 + N / 128;
                int level = static_cast<int>(plate[gi * (N + 2) + gj] / 100.0 * 9.999);
                std::cout << shades[std::clamp(level, 0, 9)];
            }
            std::cout << "|" << std::endl;
        }
    }

    MPI_Type_free(&interior);
    MPI_Type_free(&column);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
