#include <iostream>
#include <cmath>
#include <vector>
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

int main() {
    std::vector<std::vector<double>> u(N, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> u_new(N, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> f_values(N, std::vector<double>(N, 0.0));

    // Initialize f_values and set boundary conditions
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

    struct timeval t1, t2;
    gettimeofday(&t1, NULL); // Start timing

    int iteration = 0;
    double l2_diff;
    
    do {
        // Update interior points
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                u_new[i][j] = (1.0 / (2.0 * (1.0 / (dx * dx) + 1.0 / (dy * dy)))) * 
                               ((u[i-1][j] + u[i+1][j]) / (dx * dx) + 
                                (u[i][j-1] + u[i][j+1]) / (dy * dy) - 
                                f_values[i][j]);
            }
        }

        // Calculate L2 distance between current and new solution
        l2_diff = calculate_l2_distance(u, u_new);

        // Update u for the next iteration
        u = u_new;

        // Print L2 distance every 1000 iterations
        if (iteration % 1000 == 0) {
            std::cout << "Iteration " << iteration << ", L2 distance: " << l2_diff << std::endl;
        }

        iteration++;
    } while (l2_diff > convergence_threshold && iteration < max_iterations);

    gettimeofday(&t2, NULL); // End timing

    // Calculate and print the final L2 error
    double final_error = calculate_l2_error(u);

    // Output results
    std::cout << "Iterations: " << iteration << std::endl;
    std::cout << "Final L2 error: " << final_error << std::endl;
    std::cout << "Elapsed time: " 
              << ((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec))
              << " microseconds" << std::endl;

    return 0;
}

