#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>
#include "mlp.h"

// Helper to read CSV (simplified)
// Assumes first column is label, rest are pixels
void load_csv(const char* filename, std::vector<float>& X, std::vector<float>& Y, int& rows, int& cols, int max_rows = -1) {
    std::ifstream file(filename);
    std::string line;
    rows = 0;
    cols = 0;

    while (std::getline(file, line)) {
        if (max_rows > 0 && rows >= max_rows) break;
        std::stringstream ss(line);
        std::string val;
        
        // First is label
        std::getline(ss, val, ',');
        Y.push_back(std::stof(val));

        int c = 0;
        while (std::getline(ss, val, ',')) {
            X.push_back(std::stof(val) / 255.0f); // Normalize
            c++;
        }
        if (rows == 0) cols = c;
        rows++;
    }
}

// Convert labels to One-Hot
void to_one_hot(const std::vector<float>& Y, std::vector<float>& Y_one_hot, int num_classes) {
    Y_one_hot.resize(Y.size() * num_classes, 0.0f);
    for (size_t i = 0; i < Y.size(); i++) {
        int label = (int)Y[i];
        Y_one_hot[label * Y.size() + i] = 1.0f;
    }
}

int main() {
    srand(time(NULL));

    // 1. Load Data
    std::cout << "Loading data..." << std::endl;
    std::vector<float> h_X_vec, h_Y_vec;
    int samples, features;
    // Using a dummy small dataset if file not found, or try to find one.
    // For now, let's generate random data if no file.
    
    samples = 1000;
    features = 784;
    int classes = 10;
    
    h_X_vec.resize(samples * features);
    h_Y_vec.resize(samples);
    
    for (int i=0; i<samples*features; i++) h_X_vec[i] = (float)rand()/RAND_MAX;
    for (int i=0; i<samples; i++) h_Y_vec[i] = rand() % classes;

    // Transpose X to be (features, samples)
    // Currently it's (samples, features) effectively if we just filled it linearly?
    // Let's assume we want X to be column-major (features, samples).
    // So X[feature * samples + sample].
    // Our random fill is agnostic.

    std::vector<float> h_Y_one_hot;
    to_one_hot(h_Y_vec, h_Y_one_hot, classes);

    // 2. Prepare GPU Data
    float *d_X, *d_Y;
    cudaMalloc(&d_X, h_X_vec.size() * sizeof(float));
    cudaMalloc(&d_Y, h_Y_one_hot.size() * sizeof(float));

    cudaMemcpy(d_X, h_X_vec.data(), h_X_vec.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y_one_hot.data(), h_Y_one_hot.size() * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Init MLP
    std::cout << "Initializing MLP..." << std::endl;
    MLP mlp(features, 128, classes, 1); // 1 hidden layer
    mlp.init_parameters();

//—————————————————————————————————————————————— Train Loop——————————————————————————————————————————————

    // 4. Training Loop
    int epochs = 100;
    float lr = 0.1f;
    int batch_size = samples; // Full batch for simplicity

    std::cout << "Starting training..." << std::endl;
    for (int i = 0; i < epochs; i++) {
        // Forward
        mlp.forward(d_X, batch_size);

        // Backward
        mlp.backward(d_Y, batch_size);

        // Update
        // Note: We need to scale lr by 1/m here because we didn't do it in backward
        mlp.update(lr / batch_size);

        if (i % 10 == 0) {
            std::cout << "Epoch " << i << " complete." << std::endl;
            // TODO: Calculate accuracy/loss
        }
    }

    std::cout << "Training complete." << std::endl;

    cudaFree(d_X);
    cudaFree(d_Y);

    return 0;
}
