# RawGrad Backpropagation

## 1. Overview

Backpropagation is the mechanism RawGrad uses to calculate how each
trainable parameter contributed to the final loss.

RawGrad implements backpropagation explicitly rather than relying on an
automatic differentiation engine.

The complete process is:

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss
  ↓
Loss Gradient
  ↓
Backward Propagation
  ↓
Parameter Gradients
  ↓
Optimizer
  ↓
Updated Parameters