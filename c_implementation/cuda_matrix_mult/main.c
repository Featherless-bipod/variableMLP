//workflow: 
// 1.  allocate memory on the CPU
// 2.  initialize data on the CPU
// 3.  send data to the GPU
// 4.  perform the matrix multiplication on the GPU
// 5.  send the result back to the CPU
// 6.  verify the result
// 7.  free memory
//Acts as the CPU of the operation

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// links the gpu_matmul function in kernels.cu to the c compiler
void gpu_matmul(float* A, float* B, float* C, int N);

//initialize random data on CPU
void init_matrix(float* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main() {

    int N = 1024; //set up matrix size (1 million elements in total)
    size_t bytes = N * N * sizeof(float); //create matrix with N * N different elements

    printf("Allocating memory for %dx%d matrices...\n", N, N);
    float* h_A = (float*)malloc(bytes);  
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    // initialize with rando data
    srand(time(NULL));
    init_matrix(h_A, N);
    init_matrix(h_B, N);

    printf("Sending to GPU for calculation...\n");
    
    // ready setty gooooooooooo
    clock_t start = clock();
    
    // send matrices to the GPU
    gpu_matmul(h_A, h_B, h_C, N);
    
    // end timer
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Done! Time taken: %f seconds\n", time_taken);
    
    // quick verification (check the first corner element)
    printf("Verification check: C[0] = %f\n", h_C[0]);

    free(h_A); free(h_B); free(h_C); //free memory
    return 0;
}