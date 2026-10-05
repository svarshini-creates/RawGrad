# RawGrad Architecture

## 1. Overview

RawGrad is a neural-network framework implemented from scratch using
NumPy.

The project is designed to expose the internal mechanics of neural
network training rather than hiding them behind a high-level deep
learning framework.

The implementation covers:

- Dense neural-network layers
- Activation functions
- Classification losses
- Forward propagation
- Backpropagation
- Gradient descent
- Mini-batch training
- Parameter initialization
- Optimizers
- L2 regularization
- Numerical gradient checking
- MNIST classification
- Convolutional neural networks
- PyTorch validation

The core design separates model structure, mathematical operations,
optimization, and experiments into independent modules.

---

# 2. High-Level Architecture

The RawGrad training pipeline can be viewed as:

```text
                    ┌─────────────────────┐
                    │       Input X       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Sequential     │
                    │       Model         │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │             Layers              │
              │                                 │
              │ Dense / Conv2D / Flatten / Pool│
              └───────────────┬─────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────┐
              │          Activations            │
              │                                 │
              │ ReLU / Sigmoid / Softmax       │
              └───────────────┬─────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │     Predictions     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Loss          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Backpropagation  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Gradients       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Optimizer       │
                    │                     │
                    │ SGD / Momentum      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Updated Parameters  │
                    └─────────────────────┘