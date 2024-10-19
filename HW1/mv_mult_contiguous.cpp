#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <vector>
#include <algorithm> // For std::max

int main() {
    // Define the values for N and M
    std::vector<size_t> sizes = {10, 50, 100, 250, 500, 1000};

    // Print header
    std::cout << "N,M,Elapsed_Time(s),FLOPs,FLOP_Rate(GFLOPs/s),Bytes_Moved,Arithmetic_Intensity(FLOP/Byte)" << std::endl;

    for (size_t idx = 0; idx < sizes.size(); ++idx) {
        size_t N = sizes[idx];
        size_t M = sizes[idx]; // M and N vary together

        // Allocate memory for matrix A (contiguous), vectors x and y
        double* A = new double[N * M];
        double* x = new double[M];
        double* y = new double[N];

        // Initialize A and x with random values
        for (size_t i = 0; i < N * M; ++i) {
            A[i] = drand48();
        }
        for (size_t i = 0; i < M; ++i) {
            x[i] = drand48();
        }

        // Measure the time of matrix-vector multiplication
        struct timeval t_start, t_end;
        gettimeofday(&t_start, NULL);

        // Matrix-vector multiplication: y = A * x
        for (size_t i = 0; i < N; ++i) {
            y[i] = 0.0;
            for (size_t j = 0; j < M; ++j) {
                y[i] += A[i * M + j] * x[j]; // Row-major order
            }
        }

        gettimeofday(&t_end, NULL);

        // Calculate elapsed time in seconds
        double elapsed_time = (t_end.tv_sec - t_start.tv_sec) +
                              (t_end.tv_usec - t_start.tv_usec) / 1e6;

        // Calculate FLOPs and FLOP rate
        double flops = 2.0 * N * M; // One multiplication and one addition per element
        double flop_rate = flops / elapsed_time;

        // Calculate Bytes Moved
        double bytes_moved = (static_cast<double>(N) * M + M + N) * sizeof(double);

        // Calculate Arithmetic Intensity (AI)
        double arithmetic_intensity = flops / bytes_moved;

        // Output results
        std::cout << N << "," << M << "," << elapsed_time << "," << flops << "," << (flop_rate / 1e9) << "," << bytes_moved << "," << arithmetic_intensity << std::endl;

        // Clean up
        delete[] A;
        delete[] x;
        delete[] y;
    }

    return 0;
}
