#include <stdio.h>
#include <sys/time.h>
#include <cmath> // for fabs() and max()

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
#define cudaStream_t          hipStream_t
#define cudaStreamCreate      hipStreamCreate
#define cudaStreamDestroy     hipStreamDestroy
#define cudaMallocHost         hipHostMalloc
#define cudaMemcpyAsync       hipMemcpyAsync
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaFreeHost           hipHostFree
#else
#include <cuda.h>
#endif

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

__global__
void matrix_vector_multiply_chunked(double *matrix, double *vector, double *result,
                                  size_t matrix_rows, size_t vector_size,
                                  size_t start_row, size_t chunk_size) {
    // Each block handles one row of the chunk
    size_t local_row = blockIdx.x;
    size_t global_row = start_row + local_row;
    size_t tid = threadIdx.x;
    
    if (global_row >= matrix_rows || local_row >= chunk_size) return;
    
    // Shared memory for partial products within this row
    __shared__ double partial_sums[256];
    
    // Initialize shared memory
    double sum = 0.0;
    for (size_t i = tid; i < vector_size; i += blockDim.x) {
        sum += matrix[global_row * vector_size + i] * vector[i];
    }
    partial_sums[tid] = sum;
    
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        result[global_row] = partial_sums[0];
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
    
    const int NUM_STREAMS = 20;  // Number of streams to use
    const size_t CHUNK_SIZE = (matrix_rows + NUM_STREAMS - 1) / NUM_STREAMS;  // Round up division
    
    // Host pointers
    double *matrix_h, *vector_h, *result_h;
    // Device pointers
    double *matrix_d, *vector_d, *result_d;
    cudaError_t GPU_ERROR;

    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        GPU_ERROR = cudaStreamCreate(&streams[i]);
    }

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

    // Allocate pinned host memory
    GPU_ERROR = cudaMallocHost((void**)&matrix_h, sizeof(double) * matrix_rows * vector_size);
    GPU_ERROR = cudaMallocHost((void**)&vector_h, sizeof(double) * vector_size);
    GPU_ERROR = cudaMallocHost((void**)&result_h, sizeof(double) * matrix_rows);

    // Initialize host data (same as before)
    for (size_t i = 0; i < matrix_rows; i++) {
        for (size_t j = 0; j < vector_size; j++) {
            matrix_h[i * vector_size + j] = i + j;
        }
    }
    for (size_t i = 0; i < vector_size; i++) {
        vector_h[i] = 1.0;
    }

    // Allocate device memory
    GPU_ERROR = cudaMalloc((void**)&matrix_d, sizeof(double) * matrix_rows * vector_size);
    GPU_ERROR = cudaMalloc((void**)&vector_d, sizeof(double) * vector_size);
    GPU_ERROR = cudaMalloc((void**)&result_d, sizeof(double) * matrix_rows);

    // Timing variables
    double start_time = get_time();

    // Copy vector to device (needed by all chunks)
    GPU_ERROR = cudaMemcpyAsync(vector_d, vector_h, 
                               sizeof(double) * vector_size,
                               cudaMemcpyHostToDevice, streams[0]);
    GPU_ERROR = cudaStreamSynchronize(streams[0]);

    // Process chunks
    for (int i = 0; i < NUM_STREAMS; i++) {
        printf("Processing chunk %d\n", i);
        size_t start_row = i * CHUNK_SIZE;
        size_t actual_chunk_size = min(CHUNK_SIZE, matrix_rows - start_row);
        
        // Copy chunk of matrix
        GPU_ERROR = cudaMemcpyAsync(
            matrix_d + start_row * vector_size,
            matrix_h + start_row * vector_size,
            sizeof(double) * actual_chunk_size * vector_size,
            cudaMemcpyHostToDevice, 
            streams[i]
        );
        
        // Launch kernel for this chunk
        dim3 threads_per_block(256, 1, 1);
        dim3 num_blocks(actual_chunk_size, 1, 1);
        
        matrix_vector_multiply_chunked<<<num_blocks, threads_per_block, 0, streams[i]>>>(
            matrix_d, vector_d, result_d,
            matrix_rows, vector_size,
            start_row, actual_chunk_size
        );
        
        // Copy result chunk back
        GPU_ERROR = cudaMemcpyAsync(
            result_h + start_row,
            result_d + start_row,
            sizeof(double) * actual_chunk_size,
            cudaMemcpyDeviceToHost,
            streams[i]
        );
    }
    
    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        GPU_ERROR = cudaStreamSynchronize(streams[i]);
    }
    
    double end_time = get_time();
    printf("\nTotal execution time: %f seconds\n", end_time - start_time);

    // Verification code remains the same
    printf("First element of result: %f\n", result_h[0]);
    printf("Last element of result: %f\n", result_h[matrix_rows-1]);

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

    // Cleanup
    delete[] expected_result;
    
    cudaFreeHost(matrix_h);
    cudaFreeHost(vector_h);
    cudaFreeHost(result_h);
    
    cudaFree(matrix_d);
    cudaFree(vector_d);
    cudaFree(result_d);

    // Destroy streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
} 