#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <mpi.h>

// Function to calculate the exact solution
double exact_solution(double x) {
    return sin(2 * M_PI * x);  // Modify as needed
}

// Function to calculate the RHS
double rhs(double x) {
    return -4 * M_PI * M_PI * sin(2 * M_PI * x);  // Modify as needed
}

// Function to calculate the L2 norm
double calculate_l2_loss(double* u, double* u_exact, int N_local, int rank, int num_procs) {
    double local_sum = 0.0;
    for (int i = 1; i < N_local - 1; i++) {  // Ignore the ghost cells
        local_sum += (u[i] - u_exact[i]) * (u[i] - u_exact[i]);
    }
    // Remove this line:
    // fprintf(stderr, "Rank: %d, Local sum: %f\n", rank, local_sum);

    // Return the local sum instead of the square root
    return local_sum;
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    fprintf(stderr, "Rank %d of %d\n", rank, num_procs);

    int N = 100;  // Total number of grid points
    double dx = 1.0 / (N - 1);  // Grid spacing
    int max_time_steps = 1e5;
    double convergence_threshold = 1e-4;

    // Compute the number of points each process handles
    int N_local = N / num_procs + 2;  // Add ghost cells (left and right)
    
    // Allocate memory for local solution arrays
    double* u = new double[N_local];      // Numerical solution for each process
    double* u_new = new double[N_local];  // Temporary solution for the next time step
    double* u_exact = new double[N_local]; // Exact solution (used for comparison)
    double* rhs_vec = new double[N_local]; // RHS as a vector

    // Determine the global indices each process is responsible for
    int global_start = rank * (N / num_procs);
    int global_end = global_start + N_local - 2;  // Exclude ghost cells

    // Initialize local arrays
    for (int i = 1; i < N_local - 1; i++) {  
        double x = (global_start + i - 1) * dx;
        u[i] = 0.0;  // Initial condition
        u_exact[i] = exact_solution(x);  // Store the exact solution
        rhs_vec[i] = rhs(x);  // Precompute RHS at each grid point
    }

    // Apply boundary conditions (Dirichlet: u(0,t) = u(1,t) = 0)
    if (rank == 0) {
        u[1] = 0.0;
        u_new[1] = 0.0;
    }
    if (rank == num_procs - 1) {
        u[N_local - 2] = 0.0;
        u_new[N_local - 2] = 0.0;
    }

    struct timeval t1, t2;
    gettimeofday(&t1, NULL); // Start timing

    double prev_loss = calculate_l2_loss(u, u_exact, N_local, rank, num_procs);
    double current_loss = prev_loss;
    int t = 0;

    MPI_Status status;

    do {
        // Exchange boundary values with neighboring processes
        if (rank > 0) {
            MPI_Send(&u[1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&u[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
        }
        if (rank < num_procs - 1) {
            MPI_Send(&u[N_local - 2], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&u[N_local - 1], 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
        }

        // Update interior points using finite difference
        for (int i = 1; i < N_local - 1; i++) {
            u_new[i] = (rhs_vec[i] * dx * dx - u[i - 1] - u[i + 1]) / (-2);
        }

        // Swap old and new arrays
        for (int i = 1; i < N_local - 1; i++) {
            u[i] = u_new[i];
        }

        // Update previous loss
        prev_loss = current_loss;

        // Calculate current loss
        double local_loss = calculate_l2_loss(u, u_exact, N_local, rank, num_procs);
        double global_sum;
        MPI_Allreduce(&local_loss, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Calculate the global L2 norm
        current_loss = sqrt(global_sum / (N - 2));  // N - 2 to exclude global boundary points

        // Print loss at each time step (only rank 0 prints)
        if (rank == 0 && t % 100 == 0) {
            fprintf(stderr, "L2 loss at time step %d: %f\n", t, current_loss);
        }

        // Increment the time step
        t++;
    } while (fabs(prev_loss - current_loss) > convergence_threshold && t < max_time_steps);

    // Print convergence message (only rank 0 prints)
    if (rank == 0) {
        if (t < max_time_steps) {
            fprintf(stderr, "Converged after %d time steps.\n", t);
        } else {
            fprintf(stderr, "Reached maximum time steps without convergence.\n");
        }

        gettimeofday(&t2, NULL); // End timing
        fprintf(stderr, "Elapsed time: %ld microseconds\n",
                ((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec)));
        fprintf(stderr, "L2 loss: %f\n", current_loss);
    }

    // Clean up dynamically allocated memory
    delete[] u;
    delete[] u_new;
    delete[] u_exact;
    delete[] rhs_vec;

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
