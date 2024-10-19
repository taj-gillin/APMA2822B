#include <iostream>
#include <cmath>
#include <vector>
#include <mpi.h>
#include <sys/time.h>

const int N = 100;  // Number of grid points in each dimension
const double L = 1.0;  // Domain size
const double dx = L / (N - 1);
const double dy = L / (N - 1);
const int max_iterations = 1000000;
const double convergence_threshold = 1e-6;

// Function prototypes (implement these later)
double exact_solution(double x, double y);
double f(double x, double y);
double calculate_local_l2_distance(const std::vector<std::vector<double>>& u1, 
                                   const std::vector<std::vector<double>>& u2,
                                   int start_i, int end_i, int start_j, int end_j);
double calculate_local_l2_error(const std::vector<std::vector<double>>& u,
                                int start_i, int end_i, int start_j, int end_j);
double get_time();

// Function to calculate the exact solution
double exact_solution(double x, double y) {
    return sin(2 * M_PI * x) * cos(2 * M_PI * y);
}

// Function to calculate the right-hand side
double f(double x, double y) {
    return -2 * (2 * M_PI) * (2 * M_PI) * sin(2 * M_PI * x) * cos(2 * M_PI * y);
}

// Function to calculate the L2 norm of the difference between two solutions
double calculate_local_l2_distance(const std::vector<std::vector<double>>& u1, 
                                   const std::vector<std::vector<double>>& u2,
                                   int start_i, int end_i, int start_j, int end_j) {
    double sum = 0.0;
    for (int i = start_i; i <= end_i; ++i) {
        for (int j = start_j; j <= end_j; ++j) {
            double diff = u1[i][j] - u2[i][j];
            sum += diff * diff;
        }
    }
    return sum;
}

// Function to calculate the L2 error between numerical and exact solutions
double calculate_local_l2_error(const std::vector<std::vector<double>>& u,
                                int start_i, int end_i, int start_j, int end_j) {
    double sum = 0.0;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = sqrt(size);
    int cols = size / rows;
    int row = rank / cols;
    int col = rank % cols;

    int global_i_start = row * (N / rows);
    int global_j_start = col * (N / cols);

    for (int i = start_i; i <= end_i; ++i) {
        for (int j = start_j; j <= end_j; ++j) {
            double x = (global_i_start + i - 1) * dx;
            double y = (global_j_start + j - 1) * dy;
            double diff = u[i][j] - exact_solution(x, y);
            sum += diff * diff;
        }
    }
    return sum;
}

// Function to get current time in seconds
double get_time() {
    return MPI_Wtime();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Determine the 2D process grid dimensions
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int rows = dims[0];
    int cols = dims[1];

    // Create a 2D Cartesian communicator
    MPI_Comm cart_comm;
    int periods[2] = {0, 0};  // Non-periodic
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    // Get the coordinates of this process in the 2D grid
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int row = coords[0];
    int col = coords[1];

    // Calculate local domain size
    int local_N = N / rows;
    int local_M = N / cols;

    // Adjust for processes on the right and bottom edges
    if (row == rows - 1) local_N += N % rows;
    if (col == cols - 1) local_M += N % cols;

    // Allocate memory for local arrays (including ghost cells)
    std::vector<std::vector<double>> u(local_N + 2, std::vector<double>(local_M + 2, 0.0));
    std::vector<std::vector<double>> u_new(local_N + 2, std::vector<double>(local_M + 2, 0.0));
    std::vector<std::vector<double>> f_values(local_N + 2, std::vector<double>(local_M + 2, 0.0));

    // Initialize f_values and set boundary conditions
    int global_i_start = row * (N / rows);
    int global_j_start = col * (N / cols);
    for (int i = 1; i <= local_N; ++i) {
        for (int j = 1; j <= local_M; ++j) {
            double x = (global_i_start + i - 1) * dx;
            double y = (global_j_start + j - 1) * dy;
            f_values[i][j] = f(x, y);
            
            // Set boundary conditions
            if (row == 0 && i == 1) u[i][j] = u_new[i][j] = 0.0;
            if (row == rows - 1 && i == local_N) u[i][j] = u_new[i][j] = 0.0;
            if (col == 0 && j == 1) u[i][j] = u_new[i][j] = 0.0;
            if (col == cols - 1 && j == local_M) u[i][j] = u_new[i][j] = 0.0;
        }
    }

    // Main iteration loop
    int iteration = 0;
    double l2_diff;
    double start_time = get_time();

    do {
        // Exchange ghost cells with neighbors
        MPI_Request requests[8];
        MPI_Status statuses[8];
        int req_count = 0;

        // Send/Recv top edge
        if (row > 0) {
            MPI_Isend(&u[1][1], local_M, MPI_DOUBLE, rank - cols, 0, cart_comm, &requests[req_count++]);
            MPI_Irecv(&u[0][1], local_M, MPI_DOUBLE, rank - cols, 0, cart_comm, &requests[req_count++]);
        }
        // Send/Recv bottom edge
        if (row < rows - 1) {
            MPI_Isend(&u[local_N][1], local_M, MPI_DOUBLE, rank + cols, 0, cart_comm, &requests[req_count++]);
            MPI_Irecv(&u[local_N+1][1], local_M, MPI_DOUBLE, rank + cols, 0, cart_comm, &requests[req_count++]);
        }

        // Declare recv_buffers outside the if statements
        std::vector<double> left_recv_buffer(local_N);
        std::vector<double> right_recv_buffer(local_N);

        // Send/Recv left edge
        if (col > 0) {
            std::vector<double> send_buffer(local_N);
            for (int i = 1; i <= local_N; ++i) {
                send_buffer[i-1] = u[i][1];
            }
            MPI_Isend(send_buffer.data(), local_N, MPI_DOUBLE, rank - 1, 0, cart_comm, &requests[req_count++]);
            MPI_Irecv(left_recv_buffer.data(), local_N, MPI_DOUBLE, rank - 1, 0, cart_comm, &requests[req_count++]);
        }
        // Send/Recv right edge
        if (col < cols - 1) {
            std::vector<double> send_buffer(local_N);
            for (int i = 1; i <= local_N; ++i) {
                send_buffer[i-1] = u[i][local_M];
            }
            MPI_Isend(send_buffer.data(), local_N, MPI_DOUBLE, rank + 1, 0, cart_comm, &requests[req_count++]);
            MPI_Irecv(right_recv_buffer.data(), local_N, MPI_DOUBLE, rank + 1, 0, cart_comm, &requests[req_count++]);
        }

        // Wait for all communication to complete
        MPI_Waitall(req_count, requests, statuses);

        // Copy received data to ghost cells
        if (col > 0) {
            for (int i = 1; i <= local_N; ++i) {
                u[i][0] = left_recv_buffer[i-1];
            }
        }
        if (col < cols - 1) {
            for (int i = 1; i <= local_N; ++i) {
                u[i][local_M+1] = right_recv_buffer[i-1];
            }
        }

        // Update interior points
        for (int i = 1; i <= local_N; ++i) {
            for (int j = 1; j <= local_M; ++j) {
                u_new[i][j] = (1.0 / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy)))) * 
                               ((u[i-1][j] + u[i+1][j]) / (dx * dx) + 
                                (u[i][j-1] + u[i][j+1]) / (dy * dy) - 
                                f_values[i][j]);
            }
        }

        // Calculate local L2 distance
        double local_diff = calculate_local_l2_distance(u, u_new, 1, local_N, 1, local_M);
        
        // Global reduction to get sum of L2 distances
        double global_sum;
        MPI_Allreduce(&local_diff, &global_sum, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
        l2_diff = sqrt(global_sum / (N * N));

        // Update u for the next iteration
        std::swap(u, u_new);

        // Print L2 distance every 1000 iterations (only rank 0)
        if (rank == 0 && iteration % 1000 == 0) {
            std::cout << "Iteration " << iteration << ", L2 distance: " << l2_diff << std::endl;
        }

        iteration++;
    } while (l2_diff > convergence_threshold && iteration < max_iterations);

    double end_time = get_time();

    // Calculate and print the final L2 error (global)
    double local_error = calculate_local_l2_error(u, 1, local_N, 1, local_M);
    double global_error_sum;
    MPI_Reduce(&local_error, &global_error_sum, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        double final_error = sqrt(global_error_sum / (N * N));
        
        std::cout << "Iterations: " << iteration << std::endl;
        std::cout << "Final L2 error: " << final_error << std::endl;
        std::cout << "Elapsed time: " << (end_time - start_time) << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
