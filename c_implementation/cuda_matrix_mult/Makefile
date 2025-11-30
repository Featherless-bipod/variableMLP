all: #only runs first target by default
	# Compile GPU code
	nvcc -c kernels.cu -o kernels.o #NVIDIA CUDA Compiler

	# Compile CPU code
	gcc -c main.c -o main.o #GNU C Compiler

	# Link them together
	nvcc main.o kernels.o -o matrix_engine #use nvcc because can find CUDAruntime by itself
	
clean: #if want to run have to do "make clean" 
	rm -f *.o matrix_engine