clean-bins:
	rm -rf bin

clean-outs:
	rm -rf outs/*

all:
	mkdir -p bin
	make normal
	make mpi
	make omp

normal:
	g++ -o bin/hw2 src/hw2.cpp -lm

mpi:
	mpicxx -O3 -o bin/hw2_mpi src/hw2_mpi.cpp -lm

omp:
	g++ -fopenmp -O3 -o bin/hw2_omp src/hw2_omp.cpp -lm

run-mpi:
	sbatch launchers/mpi.launcher

run-normal:
	sbatch launchers/job.launcher

run-omp:
	sbatch launchers/omp.launcher

run-all:
	make run-mpi
	make run-normal
	make run-omp
