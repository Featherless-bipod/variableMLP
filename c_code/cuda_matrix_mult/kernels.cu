#include "kernels.h"
#include <stdio.h>
#include <math.h>
#include <float.h>

#define BLOCK_SIZE 16

// --- KERNELS ---

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void transpose_kernel(const float* A, float* B, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols) {
        B[c * rows + r] = A[r * cols + c];
    }
}

__global__ void relu_kernel(const float* Z, float* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmaxf(0.0f, Z[idx]);
    }
}

__global__ void apply_relu_derv_kernel(float* dZ, const float* Z, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (Z[idx] <= 0) {
            dZ[idx] = 0.0f;
        }
    }
}

// Softmax per column (batch sample)
// Assuming rows (classes) is small enough to loop over in one thread, or we use 1 block per column.
// For MNIST, rows=10. A single thread per column is fine.
__global__ void softmax_kernel(const float* Z, float* A, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        // Find max for stability
        float max_val = -FLT_MAX;
        for (int r = 0; r < rows; r++) {
            float val = Z[r * cols + col]; // Column-major access if Z is (rows, cols) stored row-major?
            // Wait, Z is stored row-major: Z[r * cols + col]
            if (val > max_val) max_val = val;
        }

        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int r = 0; r < rows; r++) {
            float val = Z[r * cols + col];
            sum_exp += expf(val - max_val);
        }

        // Normalize
        for (int r = 0; r < rows; r++) {
            float val = Z[r * cols + col];
            A[r * cols + col] = expf(val - max_val) / sum_exp;
        }
    }
}

__global__ void add_bias_kernel(float* Z, const float* b, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < rows && c < cols) {
        Z[r * cols + c] += b[r];
    }
}

__global__ void sum_cols_kernel(const float* A, float* b, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += A[r * cols + c];
        }
        b[r] = sum;
    }
}

__global__ void update_params_kernel(float* W, const float* dW, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        W[idx] -= lr * dW[idx];
    }
}


// --- WRAPPERS ---

void gpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

void gpu_transpose(const float* A, float* B, int rows, int cols) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    transpose_kernel<<<grid, block>>>(A, B, rows, cols);
}

void gpu_relu(const float* Z, float* A, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(Z, A, size);
}

void gpu_apply_relu_derv(float* dZ, const float* Z, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    apply_relu_derv_kernel<<<blocks, threads>>>(dZ, Z, size);
}

void gpu_softmax(const float* Z, float* A, int rows, int cols) {
    int threads = 256;
    int blocks = (cols + threads - 1) / threads; // One thread per column (sample)
    softmax_kernel<<<blocks, threads>>>(Z, A, rows, cols);
}

void gpu_add_bias(float* Z, const float* b, int rows, int cols) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    add_bias_kernel<<<grid, block>>>(Z, b, rows, cols);
}

void gpu_sum_cols(const float* A, float* b, int rows, int cols) {
    int threads = 256;
    int blocks = (rows + threads - 1) / threads; // One thread per row
    sum_cols_kernel<<<blocks, threads>>>(A, b, rows, cols);
}

void gpu_update_params(float* W, const float* dW, float lr, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    update_params_kernel<<<blocks, threads>>>(W, dW, lr, size);
}
