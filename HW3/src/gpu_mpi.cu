#include <stdio.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <mpi.h>

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
#define cudaStream_t          hipStream_t
#define cudaStreamCreate      hipStreamCreate
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaHostAllocDefault  hipHostMallocDefault
#define cudaMemcpyAsync       hipMemcpyAsync
#define cudaMemsetAsync       hipMemsetAsync
#else
#include <cuda.h>
#endif

// Add CUDA error checking helper
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// Add MPI error checking helper
#define MPI_CHECK(call) \
    do { \
        int err = call; \
        if (err != MPI_SUCCESS) { \
            char error_string[MPI_MAX_ERROR_STRING]; \
            int length; \
            MPI_Error_string(err, error_string, &length); \
            fprintf(stderr, "MPI error at %s:%d: %s\n", \
                    __FILE__, __LINE__, error_string); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// Constants remain the same
const int N = 200;
const double L = 1.0;
const double dx = L / (N - 1);
const double dy = L / (N - 1);
const int max_iterations = 1000000;
const double convergence_threshold = 1e-6;

// Add CUDA constant for pi
__constant__ double d_PI = 3.14159265358979323846;

// Device functions for exact solution and right-hand side
__device__ __forceinline__
double exact_solution(double x, double y) {
    return sin(2 * M_PI * x) * cos(2 * M_PI * y);
}

__device__ __forceinline__
double f(double x, double y) {
    return -2 * (2 * M_PI) * (2 * M_PI) * sin(2 * M_PI * x) * cos(2 * M_PI * y);
}

// Kernel for updating interior points
__global__
void update_solution(double* __restrict__ u_new, const double* __restrict__ u, const double* __restrict__ f_values, 
                    const int local_N, const int local_M, const double dx, const double dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i < local_N+1 && j < local_M+1) {
        int idx = i * (local_M + 2) + j;
        u_new[idx] = (1.0 / (2.0 * (1.0/(dx*dx) + 1.0/(dy*dy)))) * 
                     ((u[idx-(local_M+2)] + u[idx+(local_M+2)])/(dx*dx) + 
                      (u[idx-1] + u[idx+1])/(dy*dy) - 
                      f_values[idx]);
    }
}

// Kernel for calculating L2 norm
__global__
void calculate_l2_diff(const double* __restrict__ u1, const double* __restrict__ u2, double* __restrict__ diff_array, 
                      const int local_N, const int local_M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i < local_N+1 && j < local_M+1) {
        int idx = i * (local_M + 2) + j;
        double diff = u1[idx] - u2[idx];
        diff_array[idx] = diff * diff;
    }
}

// Add this new kernel after the existing kernels and before main():
__global__
void pack_edges(const double* __restrict__ u, 
                double* __restrict__ left_buffer,
                double* __restrict__ right_buffer,
                const int local_N, const int local_M,
                const bool pack_left, const bool pack_right) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < local_N) {
        int row = i + 1;  // Offset by 1 for ghost cells
        if (pack_left) {
            left_buffer[i] = u[row * (local_M + 2) + 1];
        }
        if (pack_right) {
            right_buffer[i] = u[row * (local_M + 2) + local_M];
        }
    }
}

int main(int argc, char** argv) {
    // Initialize MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Error: MPI implementation does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set up GPU device for this MPI rank
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus <= 0) {
        fprintf(stderr, "Error: No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Get the local rank within the node
    char *local_rank_str = getenv("SLURM_LOCALID");
    if (!local_rank_str) {
        fprintf(stderr, "Error: SLURM_LOCALID not set\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Use local rank for GPU selection to ensure proper device assignment
    int local_rank = atoi(local_rank_str);
    CUDA_CHECK(cudaSetDevice(local_rank));
    
    // Replace the print statements for GPU assignment with:
    MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks are ready
    if (rank == 0) {
        printf("Number of available GPUs: %d\n", num_gpus);
        fflush(stdout);  // Ensure output is flushed
    }
    
    // Use ordered printing for rank assignments
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            printf("Rank %d using GPU %d (local rank %d)\n", rank, local_rank, local_rank);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);  // Wait for each rank to print
    }
    MPI_Barrier(MPI_COMM_WORLD);  // Ensure all printing is done before continuing

    // Create 2D Cartesian communicator
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int rows = dims[0];
    int cols = dims[1];
    
    MPI_Comm cart_comm;
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int row = coords[0];
    int col = coords[1];

    // Calculate local domain size
    int local_N = N / rows;
    int local_M = N / cols;
    if (row == rows - 1) local_N += N % rows;
    if (col == cols - 1) local_M += N % cols;

    // Initialize CUDA streams for overlapping computation and communication
    cudaStream_t compute_stream, comm_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&comm_stream));

    // Allocate pinned host memory for better transfer performance
    double *h_u, *h_u_new, *h_f_values, *h_diff_array;
    CUDA_CHECK(hipHostMalloc((void**)&h_u, (local_N + 2) * (local_M + 2) * sizeof(double), 
                            hipHostMallocDefault));
    CUDA_CHECK(hipHostMalloc((void**)&h_u_new, (local_N + 2) * (local_M + 2) * sizeof(double), 
                            hipHostMallocDefault));
    CUDA_CHECK(hipHostMalloc((void**)&h_f_values, (local_N + 2) * (local_M + 2) * sizeof(double), 
                            hipHostMallocDefault));
    CUDA_CHECK(hipHostMalloc((void**)&h_diff_array, (local_N + 2) * (local_M + 2) * sizeof(double), 
                            hipHostMallocDefault));

    // Initialize values in pinned memory
    int global_i_start = row * (N / rows);
    int global_j_start = col * (N / cols);
    for (int i = 1; i <= local_N; ++i) {
        for (int j = 1; j <= local_M; ++j) {
            double x = (global_i_start + i - 1) * dx;
            double y = (global_j_start + j - 1) * dy;
            int idx = i * (local_M + 2) + j;
            
            h_f_values[idx] = -2 * (2 * M_PI) * (2 * M_PI) * 
                             sin(2 * M_PI * x) * cos(2 * M_PI * y);
            
            // Set boundary conditions
            if (row == 0 && i == 1 || row == rows-1 && i == local_N ||
                col == 0 && j == 1 || col == cols-1 && j == local_M) {
                h_u[idx] = sin(2 * M_PI * x) * cos(2 * M_PI * y);
                h_u_new[idx] = h_u[idx];
            }
        }
    }

    // Allocate device memory
    double *d_u, *d_u_new, *d_f_values, *d_diff_array;
    CUDA_CHECK(cudaMalloc(&d_u, (local_N + 2) * (local_M + 2) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u_new, (local_N + 2) * (local_M + 2) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f_values, (local_N + 2) * (local_M + 2) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_diff_array, (local_N + 2) * (local_M + 2) * sizeof(double)));

    // Initialize device memory
    CUDA_CHECK(cudaMemsetAsync(d_u, 0, (local_N + 2) * (local_M + 2) * sizeof(double), 
                              compute_stream));
    CUDA_CHECK(cudaMemsetAsync(d_u_new, 0, (local_N + 2) * (local_M + 2) * sizeof(double), 
                              compute_stream));

    // Copy initial data to device using streams
    CUDA_CHECK(cudaMemcpyAsync(d_u, h_u, (local_N + 2) * (local_M + 2) * sizeof(double),
                              cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_u_new, h_u_new, (local_N + 2) * (local_M + 2) * sizeof(double),
                              cudaMemcpyHostToDevice, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_f_values, h_f_values, (local_N + 2) * (local_M + 2) * sizeof(double),
                              cudaMemcpyHostToDevice, compute_stream));

    // Synchronize streams before starting computation
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    // Set up kernel configuration
    dim3 block_size(32, 32);
    dim3 grid_size(
        (local_N + block_size.x - 1) / block_size.x,
        (local_M + block_size.y - 1) / block_size.y
    );

    // Allocate buffers for edge communication
    double *send_buffer_left = new double[local_N];
    double *send_buffer_right = new double[local_N];
    double *recv_buffer_left = new double[local_N];
    double *recv_buffer_right = new double[local_N];

    // Add missing timing variable
    double start_time = MPI_Wtime();
    double end_mem;
    
    // Main iteration loop
    int iteration = 0;
    double l2_diff;
    double total_compute_time = 0.0;
    double start_mem = MPI_Wtime();

    // Add timing and performance variables
    float total_gpu_time = 0.0;  // Changed to float to match CUDA event timing
    double communication_time = 0.0;   
    double convergence_time = 0.0;     

    // Add CUDA events for precise GPU timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Add after the timing variables declaration:
    struct TimingInfo {
        double gpu_compute = 0.0;
        double communication = 0.0;
        double memory_transfer = 0.0;
        double convergence_check = 0.0;
        int iterations = 0;
    } timing;

    do {
        double iter_start = MPI_Wtime();
        
        // Time memory transfers
        double mem_start = MPI_Wtime();
        // Begin asynchronous device-to-host transfer of boundary data
        if (row > 0) {
            CUDA_CHECK(cudaMemcpyAsync(&h_u[(local_M + 2) + 1], 
                                     &d_u[(local_M + 2) + 1],
                                     local_M * sizeof(double),
                                     cudaMemcpyDeviceToHost, comm_stream));
        }
        if (row < rows - 1) {
            CUDA_CHECK(cudaMemcpyAsync(&h_u[local_N * (local_M + 2) + 1],
                                     &d_u[local_N * (local_M + 2) + 1],
                                     local_M * sizeof(double),
                                     cudaMemcpyDeviceToHost, comm_stream));
        }

        // Pack left/right edges using GPU kernel
        if (col > 0 || col < cols - 1) {
            dim3 pack_block_size(256);  // Use 1D block
            dim3 pack_grid_size((local_N + pack_block_size.x - 1) / pack_block_size.x);
            
            // Allocate device buffers for edge data if not already allocated
            double *d_send_buffer_left, *d_send_buffer_right;
            if (col > 0) {
                CUDA_CHECK(cudaMalloc(&d_send_buffer_left, local_N * sizeof(double)));
            }
            if (col < cols - 1) {
                CUDA_CHECK(cudaMalloc(&d_send_buffer_right, local_N * sizeof(double)));
            }
            
            // Pack edges using GPU
            pack_edges<<<pack_grid_size, pack_block_size, 0, comm_stream>>>(
                d_u, d_send_buffer_left, d_send_buffer_right,
                local_N, local_M, col > 0, col < cols - 1);
                
            // Copy packed data to host
            if (col > 0) {
                CUDA_CHECK(cudaMemcpyAsync(send_buffer_left, d_send_buffer_left,
                                         local_N * sizeof(double),
                                         cudaMemcpyDeviceToHost, comm_stream));
            }
            if (col < cols - 1) {
                CUDA_CHECK(cudaMemcpyAsync(send_buffer_right, d_send_buffer_right,
                                         local_N * sizeof(double),
                                         cudaMemcpyDeviceToHost, comm_stream));
            }
            
            // Free device buffers
            if (col > 0) {
                CUDA_CHECK(cudaFree(d_send_buffer_left));
            }
            if (col < cols - 1) {
                CUDA_CHECK(cudaFree(d_send_buffer_right));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(comm_stream));
        timing.memory_transfer += MPI_Wtime() - mem_start;

        // Time MPI communication
        double comm_start = MPI_Wtime();
        // Exchange ghost cells with neighbors using non-blocking MPI
        MPI_Request requests[8];
        MPI_Status statuses[8];
        int req_count = 0;

        // Send/Recv top edge
        if (row > 0) {
            MPI_CHECK(MPI_Isend(&h_u[(local_M + 2) + 1], local_M, MPI_DOUBLE, 
                               rank - cols, 0, cart_comm, &requests[req_count++]));
            MPI_CHECK(MPI_Irecv(&h_u[1], local_M, MPI_DOUBLE, 
                               rank - cols, 1, cart_comm, &requests[req_count++]));
        }

        // Send/Recv bottom edge
        if (row < rows - 1) {
            MPI_CHECK(MPI_Isend(&h_u[local_N * (local_M + 2) + 1], local_M, MPI_DOUBLE, 
                               rank + cols, 1, cart_comm, &requests[req_count++]));
            MPI_CHECK(MPI_Irecv(&h_u[(local_N + 1) * (local_M + 2) + 1], local_M, MPI_DOUBLE, 
                               rank + cols, 0, cart_comm, &requests[req_count++]));
        }

        // Send/Recv left edge (now directly from h_u)
        if (col > 0) {
            for (int i = 1; i <= local_N; ++i) {
                send_buffer_left[i-1] = h_u[i * (local_M + 2) + 1];
            }
            MPI_CHECK(MPI_Isend(send_buffer_left, local_N, MPI_DOUBLE, 
                               rank - 1, 0, cart_comm, &requests[req_count++]));
            MPI_CHECK(MPI_Irecv(recv_buffer_left, local_N, MPI_DOUBLE, 
                               rank - 1, 1, cart_comm, &requests[req_count++]));
        }

        // Send/Recv right edge (now directly from h_u)
        if (col < cols - 1) {
            for (int i = 1; i <= local_N; ++i) {
                send_buffer_right[i-1] = h_u[i * (local_M + 2) + local_M];
            }
            MPI_CHECK(MPI_Isend(send_buffer_right, local_N, MPI_DOUBLE, 
                               rank + 1, 1, cart_comm, &requests[req_count++]));
            MPI_CHECK(MPI_Irecv(recv_buffer_right, local_N, MPI_DOUBLE, 
                               rank + 1, 0, cart_comm, &requests[req_count++]));
        }
        MPI_CHECK(MPI_Waitall(req_count, requests, statuses));
        timing.communication += MPI_Wtime() - comm_start;

        // Time GPU computation
        double compute_start = MPI_Wtime();
        cudaEventRecord(start, compute_stream);
        update_solution<<<grid_size, block_size, 0, compute_stream>>>(
            d_u_new, d_u, d_f_values, local_N, local_M, dx, dy);
        cudaEventRecord(stop, compute_stream);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        timing.gpu_compute += milliseconds / 1000.0;

        // Time convergence check
        double conv_start = MPI_Wtime();
        calculate_l2_diff<<<grid_size, block_size, 0, compute_stream>>>(
            d_u_new, d_u, d_diff_array, local_N, local_M);
        // Copy difference array back to host
        CUDA_CHECK(cudaMemcpyAsync(h_diff_array, d_diff_array, 
                                 (local_N + 2) * (local_M + 2) * sizeof(double),
                                 cudaMemcpyDeviceToHost, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        // Calculate local sum
        double local_sum = 0.0;
        for (int i = 1; i <= local_N; ++i) {
            for (int j = 1; j <= local_M; ++j) {
                local_sum += h_diff_array[i * (local_M + 2) + j];
            }
        }

        // Global reduction for convergence check
        double global_sum;
        MPI_CHECK(MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, cart_comm));
        l2_diff = sqrt(global_sum / (N * N));

        // Swap pointers for next iteration
        std::swap(d_u, d_u_new);
        
        iteration++;
        timing.iterations++;
        timing.convergence_check += MPI_Wtime() - conv_start;
    } while (l2_diff > convergence_threshold && iteration < max_iterations);

    double end_time = MPI_Wtime();
    end_mem = MPI_Wtime();  // Set end_mem value that was missing

    // Calculate actual compute time from GPU events
    total_compute_time = timing.gpu_compute;  // Use the accumulated GPU timing

    // Calculate bandwidth properly
    double total_memory_accessed = iteration * (local_N * local_M) * 6 * sizeof(double); // 5 reads + 1 write per point
    double bandwidth = 0.0;
    if (total_compute_time > 0.0) {  // Avoid division by zero
        bandwidth = total_memory_accessed / (total_compute_time * 1e9); // Convert to GB/s
    }

    // Gather bandwidths from all processes
    double total_bandwidth;
    MPI_Reduce(&bandwidth, &total_bandwidth, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        std::cout << "\nResults:\n";
        std::cout << "Initial memory transfer time: " << (end_mem - start_mem) << " seconds\n";
        std::cout << "Total iterations: " << iteration << "\n";
        std::cout << "Final L2 difference: " << l2_diff << "\n";
        std::cout << "Total time: " << (end_time - start_time) << " seconds\n";
        std::cout << "Compute time: " << total_compute_time << " seconds\n";
        std::cout << "Aggregate Memory bandwidth: " << total_bandwidth << " GB/s\n";

        printf("\nDetailed Performance Breakdown:\n");
        printf("GPU Computation: %.4f seconds\n", timing.gpu_compute);
        printf("MPI Communication: %.4f seconds\n", timing.communication);
        printf("Memory Transfers: %.4f seconds\n", timing.memory_transfer);
        printf("Convergence Checks: %.4f seconds\n", timing.convergence_check);
        printf("Iterations: %d\n", timing.iterations);
        double time_per_iter = timing.iterations > 0 ? 
            (timing.gpu_compute + timing.communication + timing.memory_transfer + timing.convergence_check) / 
            timing.iterations : 0.0;
        printf("Time per iteration: %.6f seconds\n", time_per_iter);
    }

    // Cleanup
    cudaFree(d_u);
    cudaFree(d_u_new);
    cudaFree(d_f_values);
    cudaFree(d_diff_array);

    delete[] send_buffer_left;
    delete[] send_buffer_right;
    delete[] recv_buffer_left;
    delete[] recv_buffer_right;

    MPI_Finalize();
    return 0;
}

