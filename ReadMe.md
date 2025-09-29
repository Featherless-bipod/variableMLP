# 🧠 From-Scratch MLP on MNIST: Architecture Scaling and Training Analysis

## Overview
This project implements a **Multi-Layer Perceptron (MLP) from scratch in NumPy** — no PyTorch or TensorFlow — to study how architecture choices and optimization details impact training efficiency and generalization on the MNIST dataset.

The goal is not just to “get MNIST to work,” but to run **controlled experiments** and produce **Google-level analysis** of scaling trends, initialization methods, and optimization algorithms.

---

## ✨ Features
- **Pure NumPy implementation** of:
  - Forward propagation  
  - Backpropagation  
  - Gradient descent (batch + mini-batch)  
- Flexible architecture:
  - Configurable hidden layers (`H`)  
  - Configurable neurons per layer (`N`)  
- Initialization options:
  - Uniform random  
  - Xavier  
  - He (for ReLU)  
- Optimizers:
  - Vanilla SGD  
  - SGD with momentum  
  - Adam (optional extension)  
- Training instrumentation:
  - Per-epoch accuracy and loss  
  - Timing (epoch duration, total training time)  

---

## 📊 Experiments & Analysis
The core contribution of this project is the **analysis**, not just the code.  

### 1. Architecture Scaling
- **Width scaling**: vary neurons per layer (2 → 512).  
- **Depth scaling**: vary hidden layers (1 → 5).  
- **Findings**: Diminishing returns after ~128 neurons; deeper networks slowed training without accuracy gains.

### 2. Initialization
- Compared random uniform vs Xavier vs He initialization.  
- **Finding**: He initialization stabilized ReLU networks and sped up convergence.

### 3. Optimization
- Compared vanilla gradient descent, SGD with momentum, and Adam.  
- **Finding**: Adam reached high accuracy fastest; vanilla SGD lagged behind.

### 4. Batch Size
- Trained with batch sizes 1, 32, 64, 128, full batch.  
- **Finding**: Small batches improved generalization, large batches sped up training but sometimes overfit.

### 5. Pareto Analysis
- Plotted **accuracy vs training time** and **accuracy vs parameter count**.  
- Identified Pareto-optimal architectures: smallest models achieving competitive accuracy.

---

## 📈 Example Plots
- Training accuracy vs normalized epoch time (learning curves).  
- Test accuracy vs neuron count.  
- Pareto frontier: accuracy vs training time.  

---

## 🛠️ How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/mlp-from-scratch.git
   cd mlp-from-scratch


## Citation
If you use or extend this project, please credit:
@misc{yin2025mlp,
  author = {Steve Yin},
  title = {From-Scratch MLP on MNIST: Architecture Scaling and Training Analysis},
  year = {2025},
  note = {https://github.com/<Featherless-Bipod>/mlp-from-scratch}
}
