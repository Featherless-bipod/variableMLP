#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>

class MLP {
public:
    // Dimensions
    int input_dim;
    int hidden_dim;
    int output_dim;
    int num_hidden_layers;

    // Parameters (Host copies for initialization/debug, but mainly we use Device)
    // Actually, let's keep pointers to Device memory.
    // We will store them in a vector of pointers for easy iteration?
    // Or just explicit pointers for this simple 2-layer or N-layer net.
    
    // Let's support arbitrary layers.
    // Weights: W[0] is Input->H1, W[1] is H1->H2 ...
    std::vector<float*> d_W; 
    std::vector<float*> d_b;

    // Gradients
    std::vector<float*> d_dW;
    std::vector<float*> d_db;

    // Cache for Forward Pass (needed for Backprop)
    // A[0] = Input, A[1] = H1_out ...
    // Z[0] = H1_pre, Z[1] = H2_pre ...
    std::vector<float*> d_A;
    std::vector<float*> d_Z;

    // Dimensions of each layer
    std::vector<int> layer_dims; 

    MLP(int input, int hidden, int output, int layers);
    ~MLP();

    void init_parameters();
    void forward(const float* d_X, int batch_size);
    void backward(const float* d_Y, int batch_size);
    void update(float lr);
    
    // Helpers
    int get_max_dim(); // For allocating temp buffers if needed
};

#endif
