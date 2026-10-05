# RawGrad Optimizers

## 1. Overview

An optimizer is responsible for updating the trainable parameters of a
neural network using the gradients calculated during backpropagation.

RawGrad separates gradient calculation from parameter updates.

The training pipeline is:

```text
Forward Pass
     ↓
Loss
     ↓
Backpropagation
     ↓
Parameter Gradients
     ↓
Optimizer
     ↓
Updated Parameters