#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm cart_comm;
    MPI_Dims_create(argc, 1, &cart_comm);

    double local_diff = 0.0;
    double global_diff = 0.0;
    int iteration = 0;

    do {
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, cart_comm);

        iteration++;

    } while (global_diff > convergence_threshold && iteration < max_iterations);

    MPI_Finalize();

    return 0;
}
