#include <cuda_runtime.h>
#include <stdio.h>

// The __global__ keyword tells the compiler this runs on the GPU.
__global__ void matmul_kernel(float* A, float* B, float* C, int N) {
    // calculate the global row and column index for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // memory in-bounds check
    if (row < N && col < N) {
        float sum = 0.0f;
        // Each thread computes ONE single element of the result matrix
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// --- PART 2: The Wrapper (Bridge for C) ---
// extern "C" makes it so that the main.c file can find this function.
// gpu_matmul happens on the CPU
extern "C" void gpu_matmul(float* h_A, float* h_B, float* h_C, int N) {
    int size = N * N * sizeof(float);
    float *d_A, *d_B, *d_C;

    // 1. allocate vram memory on the GPU
    cudaError_t err = cudaMalloc((void**)&d_A, size); //& is to get the address of the variable
    if (err != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString(err));
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 2. copy data: host (CPU) -> device (GPU)
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); //copy data from host to device
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 3. Define Grid Dimensions
    // We break the matrix into 16x16 tiles (blocks)
    dim3 threadsPerBlock(16, 16); //choose 16*16 because standard size is 256
    // Calculate how many blocks we need to cover the whole matrix
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16); //+15 cuz it helps with rounding up

    // 4. Launch Kernel
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N); // <<<>>> is the launch configuration
    
    // Check for launch errors
    cudaDeviceSynchronize();  // cpu will wait for gpu kernal to finish
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("Kernel Error: %s\n", cudaGetErrorString(err));

    // 5. Copy Result: Device (GPU) -> Host (CPU)
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 6. Free VRAM
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}