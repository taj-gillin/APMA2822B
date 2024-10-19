#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <omp.h>

// Function to calculate the exact solution (can be manually changed)
double exact_solution(double x) {
    return sin(2 * M_PI * x);  // Change this to match your desired exact solution
}

// Function to calculate the RHS (can be manually changed, in this case it uses a sine function)
double rhs(double x) {
    return -4 * M_PI * M_PI * sin(2 * M_PI * x);  // Change this to your desired RHS
}

// Function to calculate the L2 norm (error between numerical and exact solution)
double calculate_l2_loss(double* u, double* u_exact, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        sum += (u[i] - u_exact[i]) * (u[i] - u_exact[i]);
    }
    return sqrt(sum / N);
}

int main() {
    int N = 100;  // Number of grid points
    double dx = 1.0 / (N - 1);  // Grid spacing
    int max_time_steps = 1e5;
    double convergence_threshold = 1e-3;

    // Allocate memory for the solution arrays
    double* u = new double[N];      // Numerical solution
    double* u_exact = new double[N]; // Exact solution (used for comparison)
    double* rhs_vec = new double[N]; // RHS as a vector

    // Initialize the grid (u) to zero and the exact solution array
    for (int i = 0; i < N; i++) {
        double x = i * dx;
        u[i] = 0.0;  // Initial condition: everything starts at zero
        u_exact[i] = exact_solution(x);  // Store the exact solution if known
        rhs_vec[i] = rhs(x);  // Precompute RHS at each grid point
    }

    // Apply boundary conditions (Dirichlet: u(0,t) = u(1,t) = 0)
    u[0] = 0.0;
    u[N - 1] = 0.0;

    struct timeval t1, t2;
    gettimeofday(&t1, NULL); // Start timing

    double l2_loss = calculate_l2_loss(u, u_exact, N);
    int t = 0;
    
    while (l2_loss > convergence_threshold && t < max_time_steps) {
        // Update RED points (odd indices)
        for (int i = 1; i < N - 1; i += 2) {
            u[i] = (rhs_vec[i] * dx * dx - u[i - 1] - u[i + 1]) / (-2);
        }

        // Update BLACK points (even indices)
        for (int i = 2; i < N - 1; i += 2) {
            u[i] = (rhs_vec[i] * dx * dx - u[i - 1] - u[i + 1]) / (-2);
        }

        // Print loss at each timestep
        l2_loss = calculate_l2_loss(u, u_exact, N);

        if (t % 100 == 0)
            fprintf(stderr, "L2 loss at time step %d: %f\n", t, l2_loss);

        // Increment the time step
        t++;
    }

    gettimeofday(&t2, NULL); // End timing

    // Output the elapsed time and L2 loss
    fprintf(stderr, "Elapsed time: %ld microseconds\n",
            ((t2.tv_sec * 1000000 + t2.tv_usec) - (t1.tv_sec * 1000000 + t1.tv_usec)));
    fprintf(stderr, "L2 loss: %f\n", l2_loss);
    fprintf(stderr, "Number of time steps: %d\n", t);

    // Clean up dynamically allocated memory
    delete[] u;
    delete[] u_exact;
    delete[] rhs_vec;

    return 0;
}
