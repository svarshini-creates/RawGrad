# RawGrad Experiments

## 1. Overview

This document records the verified experiments performed during the
development of RawGrad.

The experiments are designed to validate individual components,
training behavior, numerical correctness, and end-to-end functionality.

The main experiments cover:

- gradient checking
- mini-batch training
- weight initialization
- optimizer comparison
- L2 regularization
- MNIST classification
- convolutional neural networks
- PyTorch comparison

All results in this document are based on the verified RawGrad runs.

---

# 2. Experiment Categories

The experiments can be grouped into four categories:

```text
Component Validation
        ↓
Numerical Validation
        ↓
Training Experiments
        ↓
Framework Comparison