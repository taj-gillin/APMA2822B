cmake_minimum_required(VERSION 3.18)
project(poisson_solver CUDA CXX)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

# Set CUDA architectures if not already set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
endif()

add_executable(gpu_mpi src/gpu_mpi.cu)
target_link_libraries(gpu_mpi PRIVATE MPI::MPI_CXX)
set_property(TARGET gpu_mpi PROPERTY CUDA_SEPARABLE_COMPILATION ON) 