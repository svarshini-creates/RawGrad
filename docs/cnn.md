# RawGrad Convolutional Neural Network

## 1. Overview

RawGrad includes a convolutional neural network implementation built
from scratch using NumPy.

The CNN extends the framework beyond fully connected neural networks by
introducing operations that preserve and process spatial information
in images.

The implemented pipeline is:

```text
Input Image
    ↓
Conv2D
    ↓
ReLU
    ↓
MaxPool2D
    ↓
Flatten
    ↓
Dense
    ↓
Softmax
    ↓
Class Probabilities