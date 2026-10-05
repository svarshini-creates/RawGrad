# RawGrad Mathematics

## 1. Introduction

RawGrad is designed to make the mathematics behind neural-network
training explicit.

Instead of relying on automatic differentiation or a high-level
deep-learning framework, RawGrad implements the fundamental operations
using NumPy.

The main mathematical pipeline is:

```text
Input
  ↓
Linear / Convolution Operation
  ↓
Activation
  ↓
Prediction
  ↓
Loss
  ↓
Gradient
  ↓
Backpropagation
  ↓
Parameter Update