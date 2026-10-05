# RawGrad — Final Technical Report

## NumPy Neural Network Framework Built From Scratch

---

## 1. Abstract

RawGrad is a neural-network framework implemented from scratch using
Python and NumPy.

The project was developed to understand the mathematical and
computational foundations of neural-network systems rather than relying
on an existing deep-learning framework for the core implementation.

The framework progressively implements:

- Dense neural-network layers
- Activation functions
- Loss functions
- Forward propagation
- Backpropagation
- Gradient descent
- Mini-batch training
- Weight initialization
- Optimizers
- L2 regularization
- Numerical gradient checking
- MNIST classification
- Convolutional layers
- Max pooling
- Flattening
- Convolutional backpropagation

The implementation was evaluated through unit tests, numerical gradient
checks, synthetic learning experiments, MNIST classification, CNN
experiments, and selected numerical comparisons against PyTorch.

The project demonstrates that a neural-network training system can be
constructed from fundamental mathematical operations using NumPy while
remaining sufficiently modular to support both dense and convolutional
architectures.

---

# 2. Project Motivation

Modern machine-learning development commonly relies on frameworks such
as PyTorch and TensorFlow.

These frameworks provide highly optimized implementations of:

- tensor operations
- automatic differentiation
- neural-network layers
- optimizers
- GPU acceleration
- training utilities

While these tools are essential for production machine learning,
abstracting away the underlying mathematics can make it difficult to
understand what happens during model training.

RawGrad was created to address this learning gap.

Instead of beginning with a high-level training API, the project builds
the neural-network pipeline from the ground up.

The central idea is:

```text
Mathematical Operations
        ↓
Neural-Network Components
        ↓
Forward Propagation
        ↓
Loss Calculation
        ↓
Backpropagation
        ↓
Gradients
        ↓
Optimization
        ↓
Trained Model