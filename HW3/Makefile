HIPCC = hipcc
HIP_FLAGS = -x hip -O2 -DUSE_HIP -Wno-unused-result

clean-bins:
	rm -rf bin

clean-outs:
	rm -rf out/*

all:
	mkdir -p bin
	mkdir -p out
	make gpu
	make gpu-mpi

gpu:
	$(HIPCC) $(HIP_FLAGS) -o bin/gpu src/gpu.cu

gpu-mpi:
	$(HIPCC) $(HIP_FLAGS) `mpicxx --showme:compile` `mpicxx --showme:link` -o bin/gpu-mpi src/gpu_mpi.cu

run-gpu:
	sbatch launchers/gpu.launcher

run-gpu-mpi:
	sbatch launchers/gpu-mpi.launcher

run-all:
	make run-gpu
	make run-gpu-mpi
