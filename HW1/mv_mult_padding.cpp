#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <vector>
#include <algorithm> // For std::max

int main() {
    // Define the values for N and M
    std::vector<size_t> sizes = {10, 50, 100, 250, 500, 1000};

    // Assume cache line size is 64 bytes, size of double is 8 bytes
    const size_t cache_line_size = 64;
    const size_t double_size = sizeof(double);
    const size_t elements_per_cache_line = cache_line_size / double_size;

    // Print header
    std::cout << "N,M,Padded_M,Elapsed_Time(s),FLOPs,FLOP_Rate(GFLOPs/s),Bytes_Moved,Arithmetic_Intensity(FLOP/Byte)" << std::endl;

    for (size_t idx = 0; idx < sizes.size(); ++idx) {
        size_t N = sizes[idx];
        size_t M = sizes[idx]; // M and N vary together

        // Calculate padded M to align rows to cache line boundaries
        size_t padded_M = ((M + elements_per_cache_line - 1) / elements_per_cache_line) * elements_per_cache_line;

        // Allocate memory for matrix A (with padding), vectors x and y
        double* A = new double[N * padded_M];
        double* x = new double[padded_M]; // x is padded to match A's columns
        double* y = new double[N];

        // Initialize A and x with random values
        for (size_t i = 0; i < N; ++i) {
            size_t idx_base = i * padded_M;
            for (size_t j = 0; j < M; ++j) {
                A[idx_base + j] = drand48();
            }
            // Initialize padding elements to zero
            for (size_t j = M; j < padded_M; ++j) {
                A[idx_base + j] = 0.0;
            }
        }
        for (size_t i = 0; i < M; ++i) {
            x[i] = drand48();
        }
        // Initialize padding elements in x to zero
        for (size_t i = M; i < padded_M; ++i) {
            x[i] = 0.0;
        }

        // Measure the time of matrix-vector multiplication
        struct timeval t_start, t_end;
        gettimeofday(&t_start, NULL);

        // Matrix-vector multiplication: y = A * x
        for (size_t i = 0; i < N; ++i) {
            y[i] = 0.0;
            size_t idx_base = i * padded_M;
            for (size_t j = 0; j < padded_M; ++j) {
                y[i] += A[idx_base + j] * x[j];
            }
        }

        gettimeofday(&t_end, NULL);

        // Calculate elapsed time in seconds
        double elapsed_time = (t_end.tv_sec - t_start.tv_sec) +
                              (t_end.tv_usec - t_start.tv_usec) / 1e6;

        // Calculate FLOPs and FLOP rate (only count actual elements)
        double flops = 2.0 * N * M;
        double flop_rate = flops / elapsed_time;

        // Calculate Bytes Moved
        // Matrix A: N * padded_M
        // Vector x: padded_M
        // Vector y: N
        double bytes_moved = (static_cast<double>(N) * padded_M + padded_M + N) * sizeof(double);

        // Calculate Arithmetic Intensity (AI)
        double arithmetic_intensity = flops / bytes_moved;

        // Output results
        std::cout << N << "," << M << "," << padded_M << "," << elapsed_time << "," << flops << "," << (flop_rate / 1e9) << "," << bytes_moved << "," << arithmetic_intensity << std::endl;

        // Clean up
        delete[] A;
        delete[] x;
        delete[] y;
    }

    return 0;
}
