#include <stdio.h>
#include <sys/time.h>

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#define cudaGetDeviceCount     hipGetDeviceCount
#define cudaSetDevice          hipSetDevice
#define cudaDeviceSynchronize  hipDeviceSynchronize
#define cudaMalloc            hipMalloc 
#define cudaFree              hipFree
#define cudaHostMalloc        hipHostMalloc
#define cudaMemcpy            hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaError_t           hipError_t
#else
#include <cuda.h>
#endif

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

__global__
void matrix_vector_multiply(double *matrix, double *vector, double *result, 
                          size_t matrix_rows, size_t vector_size) {
    // Each block handles one row of the matrix
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;
    
    // Shared memory for partial products within this row
    __shared__ double partial_sums[256]; // Assuming max 256 threads per block
    
    // Initialize shared memory
    double sum = 0.0;
    // Each thread handles multiple elements if vector_size > blockDim.x
    for (size_t i = tid; i < vector_size; i += blockDim.x) {
        if (row < matrix_rows) {
            sum += matrix[row * vector_size + i] * vector[i];
        }
    }
    partial_sums[tid] = sum;
    
    // Synchronize to make sure all threads have written to shared memory
    __syncthreads();
    
    // Perform reduction in shared memory
    for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this row
    if (tid == 0 && row < matrix_rows) {
        result[row] = partial_sums[0];
    }
}

// Update the CPU verification function
void cpu_matrix_vector_multiply(const double *matrix, const double *vector, double *result,
                              size_t matrix_rows, size_t vector_size) {
    for (size_t i = 0; i < matrix_rows; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < vector_size; j++) {
            sum += matrix[i * vector_size + j] * vector[j];
        }
        result[i] = sum;
    }
}

int main() {
    // Define dimensions
    size_t matrix_rows = 10000;
    size_t vector_size = 10000;
    
    // Host pointers
    double *matrix_h, *vector_h, *result_h;
    // Device pointers
    double *matrix_d, *vector_d, *result_d;
    cudaError_t GPU_ERROR;

    // Check for GPU
    int ndevices = 0;
    GPU_ERROR = cudaGetDeviceCount(&ndevices);
    if (ndevices > 0) {
        printf("%d GPUs have been detected\n", ndevices);
        GPU_ERROR = cudaSetDevice(0);
    } else {
        printf("No GPUs have been detected, exiting\n");
        return 0;
    }

    // Allocate host memory
    matrix_h = new double[matrix_rows * vector_size];
    vector_h = new double[vector_size];
    result_h = new double[matrix_rows];

    // Initialize host data
    for (size_t i = 0; i < matrix_rows; i++) {
        for (size_t j = 0; j < vector_size; j++) {
            matrix_h[i * vector_size + j] = i + j;  // Simple pattern for testing
        }
    }
    for (size_t i = 0; i < vector_size; i++) {
        vector_h[i] = 1.0;  // Initialize vector with 1s
    }

    // Allocate device memory
    GPU_ERROR = cudaMalloc((void**)&matrix_d, sizeof(double) * matrix_rows * vector_size);
    GPU_ERROR = cudaMalloc((void**)&vector_d, sizeof(double) * vector_size);
    GPU_ERROR = cudaMalloc((void**)&result_d, sizeof(double) * matrix_rows);

    // Timing variables
    double start_mem, end_mem, start_compute, end_compute;
    
    // Set up execution configuration
    dim3 threads_per_block(256, 1, 1);
    dim3 num_blocks(matrix_rows, 1, 1);

    // Time memory transfers to device
    start_mem = get_time();
    GPU_ERROR = cudaMemcpy(matrix_d, matrix_h, sizeof(double) * matrix_rows * vector_size, 
                          cudaMemcpyHostToDevice);
    GPU_ERROR = cudaMemcpy(vector_d, vector_h, sizeof(double) * vector_size, 
                          cudaMemcpyHostToDevice);
    end_mem = get_time();

    // Time computation and D2H transfer
    start_compute = get_time();
    matrix_vector_multiply<<<num_blocks, threads_per_block>>>(
        matrix_d, vector_d, result_d, matrix_rows, vector_size);
    
    // Copy result back to host
    GPU_ERROR = cudaMemcpy(result_h, result_d, sizeof(double) * matrix_rows, 
                          cudaMemcpyDeviceToHost);
    GPU_ERROR = cudaDeviceSynchronize();
    end_compute = get_time();

    // Print timing results
    printf("\nTiming Results:\n");
    printf("Memory Transfer Time (H2D + D2H): %f seconds\n", end_mem - start_mem);
    printf("Computation Time (including D2H transfer): %f seconds\n", end_compute - start_compute);
    printf("Total Time: %f seconds\n\n", end_compute - start_mem);

    // Verify result (check first and last elements)
    printf("First element of result: %f\n", result_h[0]);
    printf("Last element of result: %f\n", result_h[matrix_rows-1]);

    // Verify result with CPU computation
    double *expected_result = new double[matrix_rows];
    cpu_matrix_vector_multiply(matrix_h, vector_h, expected_result, 
                             matrix_rows, vector_size);

    // Compare results
    bool passed = true;
    double max_diff = 0.0;
    for (size_t i = 0; i < matrix_rows; i++) {
        double diff = fabs(result_h[i] - expected_result[i]);
        max_diff = max(max_diff, diff);
        if (diff > 1e-10) {
            passed = false;
            printf("Mismatch at position %zu: GPU=%f, CPU=%f\n", 
                   i, result_h[i], expected_result[i]);
            break;
        }
    }
    
    printf("Verification %s\n", passed ? "PASSED" : "FAILED");
    printf("Maximum difference: %e\n", max_diff);

    // Add to cleanup
    delete[] expected_result;

    // Clean up
    delete[] matrix_h;
    delete[] vector_h;
    delete[] result_h;
    
    cudaFree(matrix_d);
    cudaFree(vector_d);
    cudaFree(result_d);

    return 0;
} 