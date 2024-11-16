
#include <stdio.h>

#include <sys/time.h>

#ifdef USE_HIP

#include <hip/hip_runtime.h>

#define cudaGetDeviceCount     hipGetDeviceCount
#define cudaSetDevice          hipSetDevice
#define cudaDeviceSynchronize  hipDeviceSynchronize


#define cudaMalloc              hipMalloc 
#define cudaFree                hipFree

#define cudaHostMalloc           hipHostMalloc
#define cudaMemcpy              hipMemcpy

#define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost

#define cudaError_t             hipError_t

#else

#include <cuda.h>

#endif

__global__
void my_dot_version1(double *x, double *y, double *results, size_t N ){

    size_t global_threadID = threadIdx.x + blockDim.x * blockIdx.x;

    //allocate shared memory for reduction within a thread bloack
    __shared__ double local_dot_result[256]; //assume up to 256 threads per block

    if (global_threadID < N){
        local_dot_result[threadIdx.x] = x[global_threadID] * y[global_threadID];
    }
    else {
        local_dot_result[threadIdx.x] = 0.0; // in case we have more theads that vector elements
    } 

    // make sure all threads completed writing data into the shared memory 
    __syncthreads();
    //reduction over shared memory

    #if 0
    //slowest version  
    if (threadIdx.x == 0){
        double sum = 0.0;
        for (int i = 0; i < blockDim.x; ++i){
            sum += local_dot_result[i];
        }
        results[blockIdx.x] = sum;
    }
    #else
    //faster version , especisally for larger blocks, but still not the fastest one
    double sum = 0.0;
    unsigned int half_group_size = blockDim.x / 2;
    for (unsigned int i = half_group_size; i > 0; i = i/2 ){
        if (threadIdx.x < i)
          local_dot_result[threadIdx.x] += local_dot_result[threadIdx.x + i];  
        __syncthreads();
    }
    if (threadIdx.x == 0) results[blockIdx.x] = local_dot_result[0]; 
    #endif

}

int main(){


    size_t N = 1000; 

    double *x_h, *y_h; //pointers to arrays on the host
    double *x_d, *y_d; //pointers to arrays on the device
    cudaError_t GPU_ERROR;

    //check if GPUs are available... if not exit
    int ndevices=0;
     GPU_ERROR = cudaGetDeviceCount(&ndevices);
    if (ndevices > 0){
      printf("%d GPUs have been detected\n",ndevices);
      GPU_ERROR = cudaSetDevice(0); //use device with ID 0 
    }
    else {
      printf("no GPUs have been detected, exiting\n");
      return 0;  
    }


    //use system memory allocator to allocate memory on the host
    x_h = new double[N];
    y_h = new double[N];

    //allocate memory on the device
    GPU_ERROR = cudaMalloc( (void**) &x_d, sizeof(double)*N);
    GPU_ERROR = cudaMalloc( (void**) &y_d, sizeof(double)*N);


    //initialize memory on the host
    //such that it will be easy to test accuracy
    for (size_t i = 0; i < N; ++i){
        x_h[i] = 0.0001*i;
        y_h[i] = 1.0;
    }

    //copy data to the device memory
    GPU_ERROR = cudaMemcpy(x_d, x_h, sizeof(double)*N, cudaMemcpyHostToDevice);
    GPU_ERROR = cudaMemcpy(y_d, y_h, sizeof(double)*N, cudaMemcpyHostToDevice);


    //how many GPU thread blocks and threads per block to use?
    dim3 nthreads(256, 1, 1);
    dim3 nblocks( (N + nthreads.x - 1) / nthreads.x, 1, 1); 

    //allocate temporary storage for intermediate results
    //call device
    double * results_hd;
    GPU_ERROR = cudaHostMalloc( (void**) &results_hd, sizeof(double)*nblocks.x);


   //call  gpu function to perform dot product and store partial results 

   my_dot_version1<<<nblocks,nthreads>>>(x_d, y_d, results_hd,N );

   //synchronize the GPU with the CPU to make sure that computation on GPU is finished
 
   GPU_ERROR = cudaDeviceSynchronize();

   //complete dot product on CPU
   double sum = 0;
   for (int i=0; i < nblocks.x; ++i){
    sum += results_hd[i];
   }

   //display results and compare to the analytical solution 

   double Sexact = (double) N * (x_h[0] + x_h[N-1])/2.0;

   //test performance of the kernel by executing it several times and averaging the execution time 
   printf(" computed dot product result = %g expected result = %g, relative error = %e\n",sum, Sexact, (sum-Sexact)/Sexact);


   //free memory
   delete[] x_h;
   delete[] y_h;

   GPU_ERROR = cudaFree(x_d);
   GPU_ERROR = cudaFree(y_d);
   return 0;
}