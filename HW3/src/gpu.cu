#include <stdio.h>
#include <cmath>
#include <sys/time.h>
#include <iostream>
#include <vector>
#include <algorithm>
#define M_PI 3.14159265358979323846

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#define cudaGetDeviceCount     hipGetDeviceCount
#define cudaSetDevice         hipSetDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMalloc           hipMalloc
#define cudaFree             hipFree
#define cudaHostMalloc       hipHostMalloc
#define cudaMemcpy           hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaError_t          hipError_t
#define cudaMemset           hipMemset
#define cudaGetErrorString   hipGetErrorString
#define cudaSuccess          hipSuccess
#define cudaGetLastError     hipGetLastError
#define cudaEvent_t          hipEvent_t
#define cudaEventCreate      hipEventCreate
#define cudaEventRecord      hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime  hipEventElapsedTime
#define cudaDeviceProp       hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#else
#include <cuda.h>
#endif

const int N = 200;  // Number of grid points in each dimension
const double L = 1.0;  // Domain size
const double dx = L / (N - 1);
const double dy = L / (N - 1);
const int max_iterations = 1000000;
const double convergence_threshold = 1e-6;

// Function to get current time in seconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Device functions for exact solution and right-hand side
__device__
double exact_solution(double x, double y) {
    return sin(2 * M_PI * x) * cos(2 * M_PI * y);
}

__device__
double f(double x, double y) {
    return -2 * (2 * M_PI) * (2 * M_PI) * sin(2 * M_PI * x) * cos(2 * M_PI * y);
}

// Kernel for updating interior points
__global__
void update_solution(double* u_new, const double* u, const double* f_values, 
                    int N, double dx, double dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i < N-1 && j < N-1) {
        int idx = i * N + j;
        u_new[idx] = (1.0 / (2.0 * (1.0/(dx*dx) + 1.0/(dy*dy)))) * 
                     ((u[idx-N] + u[idx+N])/(dx*dx) + 
                      (u[idx-1] + u[idx+1])/(dy*dy) - 
                      f_values[idx]);
    }
}

// Kernel for calculating L2 norm
__global__
void calculate_l2_diff(const double* u1, const double* u2, double* diff_array, 
                      int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i < N-1 && j < N-1) {
        int idx = i * N + j;
        double diff = u1[idx] - u2[idx];
        diff_array[idx] = diff * diff;
    }
}

// Kernel for calculating L2 error against exact solution
__global__
void calculate_l2_error(const double* u, double* error_array, 
                       int N, double dx, double dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i < N-1 && j < N-1) {
        int idx = i * N + j;
        double x = i * dx;
        double y = j * dy;
        double diff = u[idx] - exact_solution(x, y);
        error_array[idx] = diff * diff;
    }
}

int main() {
    // Check for CUDA device
    int ndevices;
    cudaGetDeviceCount(&ndevices);
    if (ndevices == 0) {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }
    
    // Allocate host memory
    std::vector<double> h_u(N * N, 0.0);
    std::vector<double> h_u_new(N * N, 0.0);
    std::vector<double> h_f_values(N * N, 0.0);
    
    // Initialize boundary conditions and f_values on host
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double x = i * dx;
            double y = j * dy;
            
            if (i == 0 || i == N-1 || j == 0 || j == N-1) {
                h_u[i*N + j] = sin(2 * M_PI * x) * cos(2 * M_PI * y);
                h_u_new[i*N + j] = h_u[i*N + j];
            }
            h_f_values[i*N + j] = -2 * (2 * M_PI) * (2 * M_PI) * 
                                 sin(2 * M_PI * x) * cos(2 * M_PI * y);
        }
    }
    
    // Allocate device memory
    double *d_u, *d_u_new, *d_f_values, *d_diff_array;
    cudaMalloc(&d_u, N * N * sizeof(double));
    cudaMalloc(&d_u_new, N * N * sizeof(double));
    cudaMalloc(&d_f_values, N * N * sizeof(double));
    cudaMalloc(&d_diff_array, N * N * sizeof(double));
    
    // Copy data to device
    double start_mem = get_time();
    cudaMemcpy(d_u, h_u.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_new, h_u_new.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_values, h_f_values.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
    double end_mem = get_time();
    
    // Set up kernel configuration
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
                   (N + block_size.y - 1) / block_size.y);
    
    // Reset timing variables
    double compute_time = 0.0;        // Time spent in kernels
    double communication_time = 0.0;   // Time spent in memory transfers
    double convergence_time = 0.0;     // Time spent checking convergence
    
    double start_time = get_time();
    
    // Main iteration loop
    int iteration = 0;
    double l2_diff;
    std::vector<double> h_diff_array(N * N);
    
    // Replace gettimeofday with CUDA events for more accurate GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    do {
        // Time the main computation kernel
        cudaEventRecord(start);
        update_solution<<<grid_size, block_size>>>(d_u_new, d_u, d_f_values, N, dx, dy);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        compute_time += milliseconds / 1000.0;

        // Time the convergence check (includes both kernel and memory transfer)
        double conv_start = get_time();
        calculate_l2_diff<<<grid_size, block_size>>>(d_u_new, d_u, d_diff_array, N);
        cudaDeviceSynchronize();
        
        // Time just the communication part
        double comm_start = get_time();
        cudaMemcpy(h_diff_array.data(), d_diff_array, N * N * sizeof(double), 
                   cudaMemcpyDeviceToHost);
        double comm_end = get_time();
        
        communication_time += (comm_end - comm_start);
        convergence_time += (comm_end - conv_start);

        double sum = 0.0;
        for (int i = 1; i < N-1; ++i) {
            for (int j = 1; j < N-1; ++j) {
                sum += h_diff_array[i*N + j];
            }
        }
        l2_diff = sqrt(sum / ((N-2) * (N-2)));
        
        // Swap pointers
        std::swap(d_u, d_u_new);
        
        if (iteration % 1000 == 0) {
            std::cout << "Iteration " << iteration << ", L2 distance: " << l2_diff 
                      << ", Time: " << (milliseconds / 1000.0) << " seconds\n";
        }
        
        iteration++;
    } while (l2_diff > convergence_threshold && iteration < max_iterations);
    
    double end_time = get_time();
    
    // Calculate final error
    calculate_l2_error<<<grid_size, block_size>>>(d_u, d_diff_array, N, dx, dy);
    cudaMemcpy(h_diff_array.data(), d_diff_array, N * N * sizeof(double), 
               cudaMemcpyDeviceToHost);
    
    double error_sum = 0.0;
    for (int i = 1; i < N-1; ++i) {
        for (int j = 1; j < N-1; ++j) {
            error_sum += h_diff_array[i*N + j];
        }
    }
    double final_error = sqrt(error_sum / ((N-2) * (N-2)));
    
    // Calculate bandwidth using the actual compute time
    double total_memory_accessed = iteration * ((N-2) * (N-2)) * 7 * sizeof(double);
    
    // Convert to GB/s (divide by 1e9 for GB and by compute_time for per-second rate)
    double bandwidth = (total_memory_accessed / 1e9) / compute_time;
    
    // Print results with more detailed timing information
    std::cout << "\nResults:\n";
    std::cout << "Initial memory transfer time: " << (end_mem - start_mem) << " seconds\n";
    std::cout << "Total iterations: " << iteration << "\n";
    std::cout << "Final L2 error: " << final_error << "\n";
    std::cout << "Total time: " << (end_time - start_time) << " seconds\n";
    std::cout << "Compute time: " << compute_time << " seconds\n";
    std::cout << "Memory bandwidth: " << bandwidth << " GB/s\n";
    
    // Cleanup
    cudaFree(d_u);
    cudaFree(d_u_new);
    cudaFree(d_f_values);
    cudaFree(d_diff_array);
    
    // Main iteration kernel:
    double flops_per_point = 12;  // Updated from 11
    double mem_ops_per_point = 6; // Updated from 7
    double main_flops = iteration * (N-2) * (N-2) * flops_per_point;
    double main_bytes = iteration * (N-2) * (N-2) * mem_ops_per_point * sizeof(double);
    double main_ai = (double)main_flops / main_bytes;
    double main_bandwidth = main_bytes / (compute_time * 1e9); // GB/s
    double main_flops_per_sec = main_flops / compute_time;

    // Convergence check:
    double conv_flops = iteration * (N-2) * (N-2) * 3;  // 3 FLOPS per point
    double conv_bytes = iteration * (N-2) * (N-2) * 2 * sizeof(double);
    double conv_ai = (double)conv_flops / conv_bytes;
    double conv_bandwidth = conv_bytes / (convergence_time * 1e9);
    double conv_flops_per_sec = conv_flops / convergence_time;

    // Print detailed performance metrics
    std::cout << "\nPerformance Metrics:\n";
    std::cout << "Main Iteration Kernel:\n";
    std::cout << "  Arithmetic Intensity: " << main_ai << " FLOPS/byte\n";
    std::cout << "  Bandwidth: " << main_bandwidth << " GB/s\n";
    std::cout << "  FLOPS: " << main_flops_per_sec << " FLOPS/s\n";
    std::cout << "  Time: " << compute_time << " seconds\n";
    
    std::cout << "\nConvergence Check:\n";
    std::cout << "  Arithmetic Intensity: " << conv_ai << " FLOPS/byte\n";
    std::cout << "  Bandwidth: " << conv_bandwidth << " GB/s\n";
    std::cout << "  FLOPS: " << conv_flops_per_sec << " FLOPS/s\n";
    std::cout << "  Time: " << convergence_time << " seconds\n";
    
    std::cout << "\nCommunication:\n";
    std::cout << "  Time: " << communication_time << " seconds\n";
    
    std::cout << "Debug Info:\n";
    std::cout << "  Points processed per iteration: " << ((N-2) * (N-2)) << "\n";
    std::cout << "  Total bytes accessed: " << main_bytes << "\n";
    std::cout << "  Compute time (s): " << compute_time << "\n";
    
    // Try calculating bandwidth based on total data movement
    double total_elements = iteration * ((N-2) * (N-2));
    double total_bytes = total_elements * (7 * sizeof(double));
    
    std::cout << "Achieved Bandwidth: " << main_bandwidth << " GB/s\n";
    
    // Make sure we're using the correct units
    double bandwidth_GB_s = (main_bytes / (1024.0 * 1024.0 * 1024.0)) / compute_time;
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Peak theoretical bandwidth (GB/s)
    double peak_bandwidth = prop.memoryClockRate * 1000.0 * 
                           (prop.memoryBusWidth / 8.0) * 2.0 / 1.0e9;

    // Peak theoretical FLOPS (GFLOPS)
    double peak_flops = prop.clockRate * 1000.0 * 
                       prop.multiProcessorCount * 
                       prop.warpSize * 2.0 / 1.0e9;  // Assuming 2 ops per clock

    // Roofline intersection point
    double ridge_point = peak_flops / peak_bandwidth;

    // Actual performance metrics
    double achieved_flops = main_flops_per_sec / 1.0e9;  // GFLOPS
    double achieved_bandwidth = main_bandwidth;           // GB/s
    double achieved_ai = main_ai;

    std::cout << "\nRoofline Model Metrics:\n";
    std::cout << "Peak Memory Bandwidth: " << peak_bandwidth << " GB/s\n";
    std::cout << "Peak Compute: " << peak_flops << " GFLOPS\n";
    std::cout << "Ridge Point: " << ridge_point << " FLOPS/byte\n";
    std::cout << "Achieved Arithmetic Intensity: " << achieved_ai << " FLOPS/byte\n";
    std::cout << "Memory or Compute Bound: " << 
                (achieved_ai < ridge_point ? "Memory" : "Compute") << " bound\n";

    return 0;
}