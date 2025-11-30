#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

// Matrix Multiplication: C = A * B
// A: (M x K), B: (K x N), C: (M x N)
void gpu_matmul(const float* A, const float* B, float* C, int M, int N, int K);

// Matrix Transpose: B = A^T
// A: (rows x cols), B: (cols x rows)
void gpu_transpose(const float* A, float* B, int rows, int cols);

// Element-wise ReLU: A = max(0, Z)
void gpu_relu(const float* Z, float* A, int size);

// Apply ReLU Derivative: dZ = dZ * (Z > 0)
void gpu_apply_relu_derv(float* dZ, const float* Z, int size);

// Softmax (Column-wise): A = softmax(Z)
// rows: number of classes, cols: batch size
void gpu_softmax(const float* Z, float* A, int rows, int cols);

// Add Bias: Z = Z + b (broadcast b to every column)
// Z: (rows x cols), b: (rows x 1)
void gpu_add_bias(float* Z, const float* b, int rows, int cols);

// Sum Columns: b = sum(A, axis=1)
// A: (rows x cols), b: (rows x 1)
void gpu_sum_cols(const float* A, float* b, int rows, int cols);

// Parameter Update: W = W - lr * dW
void gpu_update_params(float* W, const float* dW, float lr, int size);

#endif
