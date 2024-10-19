#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <algorithm>
#include <vector>

// Function to calculate the exact solution (can be manually changed)
double exact_solution(double x) {
    return sin(2 * M_PI * x);  // Change this to match your desired exact solution
}

// Function to calculate the RHS (can be manually changed, in this case it uses a sine function)
double rhs(double x) {
    return -4 * M_PI * M_PI * sin(2 * M_PI * x);  // Change this to your desired RHS
}

// Function to calculate the L2 norm (error between numerical and exact solution)
double calculate_l2_loss(const std::vector<double>& u, const std::vector<double>& u_exact) {
    double sum = 0.0;
    for (size_t i = 0; i < u.size(); i++) {
        sum += (u[i] - u_exact[i]) * (u[i] - u_exact[i]);
    }
    return sqrt(sum / u.size());
}

// Function to calculate the L2 distance between two vectors
double calculate_l2_distance(const std::vector<double>& u1, const std::vector<double>& u2) {
    double sum = 0.0;
    for (size_t i = 0; i < u1.size(); i++) {
        sum += (u1[i] - u2[i]) * (u1[i] - u2[i]);
    }
    return sqrt(sum / u1.size());
}

int main() {
    const int N = 1000;  // Number of grid points (increased for better accuracy)
    const double dx = 1.0 / (N - 1);  // Grid spacing
    const int max_time_steps = 1e6;  // Increased max time steps
    const double convergence_threshold = 1e-8;  // Decreased threshold for better convergence

    // Use std::vector for dynamic memory management
    std::vector<double> u(N, 0.0);      // Numerical solution
    std::vector<double> u_new(N, 0.0);  // Temporary solution for the next time step
    std::vector<double> u_exact(N);     // Exact solution (used for comparison)
    std::vector<double> rhs_vec(N);     // RHS as a vector

    // Initialize the grid (u) to zero and the exact solution array
    for (int i = 0; i < N; i++) {
        double x = i * dx;
        u_exact[i] = exact_solution(x);  // Store the exact solution if known
        rhs_vec[i] = rhs(x);  // Precompute RHS at each grid point
    }

    // Apply boundary conditions (Dirichlet: u(0,t) = u(1,t) = 0)
    u[0] = u[N - 1] = u_new[0] = u_new[N - 1] = 0.0;

    struct timeval t1, t2;
    gettimeofday(&t1, NULL); // Start timing

    double l2_diff;
    int t = 0;

    do {
        // Update the interior points using the finite difference method
        for (int i = 1; i < N - 1; i++) {
            u_new[i] = ((dx * dx) * rhs_vec[i] - u[i-1] - u[i+1]) / (-2);
        }

        // Calculate L2 distance between current and new solution
        l2_diff = calculate_l2_distance(u, u_new);

        // Update u for the next iteration
        u = u_new;

        // Print L2 distance at each timestep
        if (t % 1000 == 0)
            std::cerr << "L2 distance at time step " << t << ": " << l2_diff << std::endl;

        // Increment the time step
        t++;

    } while (l2_diff > convergence_threshold && t < max_time_steps);

    if (t < max_time_steps) {
        std::cerr << "Converged after " << t << " time steps." << std::endl;
    } else {
        std::cerr << "Reached maximum time steps without convergence." << std::endl;
    }

    gettimeofday(&t2, NULL); // End timing

    // Output the elapsed time and L2 loss
    std::cerr << "Elapsed time: " 
              << ((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec))
              << " microseconds" << std::endl;

    // Calculate final L2 loss for reporting
    double final_loss = calculate_l2_loss(u, u_exact);

    std::cerr << "Final L2 loss: " << final_loss << std::endl;
    std::cerr << "Number of time steps: " << t << std::endl;

    return 0;
}
