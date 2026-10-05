# RawGrad MNIST Classification

## 1. Overview

MNIST is the main image-classification benchmark used in RawGrad to
demonstrate that the framework can train a neural network on a real
dataset.

The experiment uses the MNIST handwritten-digit dataset and implements
the complete classification pipeline with RawGrad.

The pipeline is:

```text
MNIST IDX Files
      ↓
Dataset Loading
      ↓
Image Normalization
      ↓
Flatten 28 × 28 Images
      ↓
Dense Neural Network
      ↓
Softmax
      ↓
Categorical Cross-Entropy
      ↓
Backpropagation
      ↓
Mini-Batch SGD
      ↓
Evaluation