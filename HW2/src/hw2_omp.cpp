#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <sys/time.h>

const int N = 100;  // Number of grid points in each dimension
const double L = 1.0;  // Domain size
const double dx = L / (N - 1);
const double dy = L / (N - 1);
const int max_iterations = 1000000;
const double convergence_threshold = 1e-6;

// Function to calculate the exact solution
double exact_solution(double x, double y) {
    return sin(2 * M_PI * x) * cos(2 * M_PI * y);
}

// Function to calculate the right-hand side
double f(double x, double y) {
    return -2 * (2 * M_PI) * (2 * M_PI) * sin(2 * M_PI * x) * cos(2 * M_PI * y);
}

// Function to calculate the L2 norm of the difference between two solutions
double calculate_l2_distance(const std::vector<std::vector<double>>& u1, 
                             const std::vector<std::vector<double>>& u2) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) collapse(2)
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            double diff = u1[i][j] - u2[i][j];
            sum += diff * diff;
        }
    }
    return sqrt(sum / ((N - 2) * (N - 2)));
}

// Function to calculate the L2 error between numerical and exact solutions
double calculate_l2_error(const std::vector<std::vector<double>>& u) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) collapse(2)
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            double x = i * dx;
            double y = j * dy;
            double diff = u[i][j] - exact_solution(x, y);
            sum += diff * diff;
        }
    }
    return sqrt(sum / ((N - 2) * (N - 2)));
}

// Function to get current time in seconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    std::cout << "Maximum number of threads: " << omp_get_max_threads() << std::endl;

    // Test parallel execution
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Number of threads in use: " << omp_get_num_threads() << std::endl;
        
        #pragma omp critical
        std::cout << "Hello from thread " << omp_get_thread_num() << std::endl;
    }

    std::vector<std::vector<double>> u(N, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> u_new(N, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> f_values(N, std::vector<double>(N, 0.0));

    // Initialize f_values and set boundary conditions
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double x = i * dx;
            double y = j * dy;
            f_values[i][j] = f(x, y);
            
            // Set boundary conditions
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
                u[i][j] = u_new[i][j] = 0.0;
            }
        }
    }

    double start_time = get_time();
    double parallel_time = 0.0;

    int iteration = 0;
    double l2_diff;
    
    do {
        double iteration_start_time = get_time();

        // Update interior points
        double loop_start_time = get_time();
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                u_new[i][j] = (1.0 / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy)))) * 
                               ((u[i-1][j] + u[i+1][j]) / (dx * dx) + 
                                (u[i][j-1] + u[i][j+1]) / (dy * dy) - 
                                f_values[i][j]);
            }
        }
        double loop_end_time = get_time();
        parallel_time += loop_end_time - loop_start_time;

        // Calculate L2 distance between current and new solution
        l2_diff = calculate_l2_distance(u, u_new);

        // Update u for the next iteration
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                u[i][j] = u_new[i][j];
            }
        }

        double iteration_end_time = get_time();
        double iteration_time = iteration_end_time - iteration_start_time;

        // Print L2 distance and iteration time every 1000 iterations
        if (iteration % 1000 == 0) {
            std::cout << "Iteration " << iteration << ", L2 distance: " << l2_diff 
                      << ", Time: " << iteration_time << " seconds" << std::endl;
        }

        iteration++;
    } while (l2_diff > convergence_threshold && iteration < max_iterations);

    double end_time = get_time();

    // Calculate and print the final L2 error
    double final_error = calculate_l2_error(u);

    // Calculate bandwidth for main computation loop
    double total_memory_accessed = iteration * (N - 2) * (N - 2) * 6 * sizeof(double);  // 5 reads, 1 write per point
    double bandwidth = total_memory_accessed / (end_time - start_time) / 1e9;  // GB/s

    // Output results
    std::cout << "Iterations: " << iteration << std::endl;
    std::cout << "Final L2 error: " << final_error << std::endl;
    std::cout << "Elapsed time: " << (end_time - start_time) << " seconds" << std::endl;
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
    std::cout << "Total time in parallel region: " << parallel_time << " seconds" << std::endl;
    std::cout << "Percentage of time in parallel: " << (parallel_time / (end_time - start_time)) * 100 << "%" << std::endl;

    return 0;
}
