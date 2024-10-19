clean-bins:
	rm -rf bin

clean-outs:
	rm -rf outs/*

build:
	mkdir -p bin
	mpicxx -o bin/pde_mpi pde_mpi.cpp -lm
	g++ -o bin/pde pde.cpp -lm
	g++ -o bin/rbparallel rbparallel.cpp -lm
	g++ -o bin/redblack redblack.cpp -lm	

run-mpi:
	sbatch launchers/mpijob.launcher

run-normal:
	sbatch launchers/job.launcher
