HIPCC = hipcc
HIP_FLAGS = -x hip -O3 -DUSE_HIP -Wno-unused-result

clean-bins:
	rm -rf bin

clean-outs:
	rm -rf outs/*

all:
	mkdir -p bin
	make vector-matrix
	make vm-memstream
	make dot

vector-matrix:
	$(HIPCC) $(HIP_FLAGS) -o bin/vector_matrix_mul src/vector_matrix_mul.cu

vm-memstream:
	$(HIPCC) $(HIP_FLAGS) -o bin/vm_memstream src/vm_memstream.cu

dot:
	$(HIPCC) $(HIP_FLAGS) -o bin/dot src/dot.cu

run-vector-matrix:
	./bin/vector_matrix_mul

run-vm-memstream:
	./bin/vm_memstream

run-dot:
	./bin/dot

run-all:
	make run-vector-matrix
	make run-vm-memstream
	make run-dot
