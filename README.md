# RawGrad

### A NumPy Neural Network Framework Built From Scratch

RawGrad is a neural-network framework implemented from first principles using Python and NumPy.

The goal is simple:

> Understand what happens inside a neural network by building the machinery yourself.

Instead of relying on automatic differentiation or a high-level deep-learning framework for the core implementation, RawGrad implements the major pieces of neural-network training explicitly:

- layers
- activations
- losses
- forward propagation
- backpropagation
- gradient descent
- mini-batch training
- optimizers
- regularization
- gradient checking
- MNIST classification
- convolutional neural networks

The implementation is validated through unit tests, numerical gradient checking, real-dataset experiments, CNN experiments, and selected numerical comparisons against PyTorch.

---

## Why RawGrad?

Deep-learning frameworks make model development extremely productive, but they also hide much of the mathematics behind abstractions.

RawGrad takes the opposite approach.

The project builds the training pipeline step by step:

```text
Input
  ↓
Layer
  ↓
Activation
  ↓
Prediction
  ↓
Loss
  ↓
Loss Gradient
  ↓
Backpropagation
  ↓
Parameter Gradients
  ↓
Optimizer
  ↓
Updated Parameters
```

The framework is intentionally small and transparent so that the mathematical relationship between forward propagation, loss computation, gradients, and parameter updates remains visible.

---

## Features

### Core Neural Network Components

- Dense layers
- ReLU activation
- Sigmoid activation
- Softmax activation
- Mean Squared Error
- Binary Cross-Entropy
- Categorical Cross-Entropy
- Sequential models
- Forward propagation
- Backpropagation

### Training

- Full-batch gradient descent
- Mini-batch training
- Train/test evaluation
- Weight initialization
  - Standard
  - Xavier
  - He
- SGD optimizer
- Momentum optimizer

### Regularization and Validation

- L2 regularization
- Numerical gradient checking
- Analytical vs numerical gradient comparison
- Unit tests for core components
- Shape and numerical validation

### Computer Vision

- MNIST dataset loading
- Dense MNIST classifier
- 2D convolution
- ReLU
- Max pooling
- Flattening
- CNN classification pipeline

### Framework Validation

- Numerical gradient checking
- CNN parameter-update validation
- Selected numerical comparison against PyTorch

---

## Project Roadmap

RawGrad was developed through the following 16 milestones:

| Milestone | Topic | Status |
|---|---|---|
| 1 | Single neuron | Complete |
| 2 | Multiple neurons | Complete |
| 3 | Dense layer | Complete |
| 4 | Activation functions | Complete |
| 5 | Loss functions | Complete |
| 6 | Forward propagation | Complete |
| 7 | Backpropagation | Complete |
| 8 | Gradient descent | Complete |
| 9 | Mini-batch training | Complete |
| 10 | Optimizers | Complete |
| 11 | Regularization | Complete |
| 12 | Gradient checking | Complete |
| 13 | MNIST classification | Complete |
| 14 | CNN from scratch | Complete |
| 15 | PyTorch comparison | Complete |
| 16 | Final RawGrad framework | Complete |

---

## Architecture

The main framework is organized around a small set of components:

```text
rawgrad/
│
├── activations.py
├── classification_losses.py
├── convolution.py
├── flatten.py
├── layers.py
├── losses.py
├── mnist.py
├── model.py
├── optimizers.py
└── pooling.py
```

### High-Level Architecture

```text
                 RawGrad
                    │
          ┌─────────┴─────────┐
          │                   │
       Layers              Models
          │                   │
   ┌──────┴──────┐       Sequential
   │             │
 Dense        Conv2D
   │             │
   └──────┬──────┘
          │
     Activations
          │
   ┌──────┼─────────┐
   │      │         │
 ReLU  Sigmoid   Softmax
          │
        Loss
          │
   ┌──────┴──────────┐
   │                 │
Loss Gradient   Regularization
   │
Backpropagation
   │
Parameter Gradients
   │
Optimizers
   │
Updated Parameters
```

---

## Installation

Clone the repository and create a virtual environment.

### Windows

```powershell
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd RawGrad

py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install RawGrad:

```powershell
python -m pip install -e .
```

Install the testing dependency:

```powershell
python -m pip install pytest
```

RawGrad's core implementation uses NumPy.

---

## Quick Example

```python
import numpy as np

from rawgrad.layers import Dense
from rawgrad.activations import ReLU, Softmax
from rawgrad.model import Sequential


X = np.random.randn(8, 4)

model = Sequential([
    Dense(4, 8, initialization="xavier"),
    ReLU(),
    Dense(8, 3, initialization="xavier"),
    Softmax(),
])

predictions = model.forward(X)

print(predictions.shape)
```

Output:

```text
(8, 3)
```

The model can then participate in the same forward, loss, backward, and optimization pipeline implemented throughout RawGrad.

---

## Training Pipeline

A typical RawGrad training step follows this sequence:

```text
1. Input batch
       ↓
2. Forward propagation
       ↓
3. Prediction
       ↓
4. Loss calculation
       ↓
5. Loss gradient
       ↓
6. Backpropagation
       ↓
7. Parameter gradients
       ↓
8. Optimizer update
       ↓
9. Next batch
```

This process is implemented explicitly rather than delegated to an automatic differentiation engine.

---

## Backpropagation

Backpropagation is implemented by propagating gradients backward through each layer.

For a dense layer:

```text
Input
  ↓
W, b
  ↓
Z = XW + b
  ↓
Activation
  ↓
Output
```

During the backward pass, RawGrad calculates gradients with respect to the layer parameters and input.

For a dense layer:

```text
dW = Xᵀ dZ

db = Σ dZ

dX = dZ Wᵀ
```

The gradients are then passed to the previous layer.

More details are available in:

- `docs/backpropagation.md`
- `docs/mathematics.md`

---

## Gradient Descent

RawGrad implements parameter optimization explicitly.

A basic gradient-descent update is:

```text
parameter ← parameter - learning_rate × gradient
```

The project also includes:

- SGD
- Momentum

The optimizer implementations can be found in:

```text
src/rawgrad/optimizers.py
```

---

## Mini-Batch Training

RawGrad supports mini-batch training instead of requiring the complete dataset to be processed in a single update.

The training process becomes:

```text
Dataset
   ↓
Shuffle
   ↓
Mini-batch
   ↓
Forward
   ↓
Loss
   ↓
Backward
   ↓
Optimizer update
   ↓
Next mini-batch
```

This provides a more realistic training workflow while keeping the implementation transparent.

---

## Weight Initialization

RawGrad supports multiple initialization strategies:

- Standard initialization
- Xavier initialization
- He initialization

The project also includes an experiment comparing these strategies with zero initialization.

The experiment demonstrates why appropriate initialization matters when training neural networks.

See:

```text
examples/21_weight_initialization.py
```

and:

```text
docs/experiments.md
```

---

## Optimizers

RawGrad currently implements:

### SGD

Standard stochastic gradient descent.

### Momentum

Momentum maintains a velocity term to smooth parameter updates and accelerate optimization in suitable settings.

An optimizer comparison is available in:

```text
examples/22_optimizer_comparison.py
```

The project also includes a dedicated notebook:

```text
notebooks/04_optimizers.ipynb
```

---

## L2 Regularization

RawGrad supports L2 regularization on trainable weights.

The regularization penalty is:

```text
L₂ = ½ λ ||W||²
```

The corresponding gradient contribution is:

```text
λW
```

Bias parameters are not regularized.

The implementation is available through the dense layers and model-level regularization loss.

Example:

```text
examples/23_l2_regularization.py
```

Documentation:

```text
docs/regularization.md
```

---

## Gradient Checking

One of the most important validation mechanisms in RawGrad is numerical gradient checking.

The analytical gradient produced by backpropagation is compared against a numerical approximation.

The numerical approximation uses finite differences:

```text
f(θ + ε) - f(θ - ε)
--------------------
        2ε
```

The implemented gradient checker verifies that the two calculations agree within a specified tolerance.

Verified result:

```text
Parameters checked : 10
Maximum absolute error : 1.016e-11
Maximum relative error : 5.634e-09
Tolerance : 1e-05

Result: PASS
```

Example:

```text
examples/19_gradient_checking.py
```

Documentation:

```text
docs/gradient_checking.md
```

---

## MNIST Classification

RawGrad includes an MNIST data loader and a dense neural-network classification pipeline.

The loader supports the original IDX gzip files and converts the images into normalized NumPy arrays.

Dataset shapes:

```text
Training images : 60,000 × 784
Test images     : 10,000 × 784
```

The demonstration training experiment uses a smaller subset for faster execution.

### Verified Dense MNIST Experiment

Configuration:

```text
Training samples : 5,000
Testing samples  : 1,000
Input features   : 784
Classes          : 10
Epochs           : 5
Batch size       : 64
Optimizer        : SGD
Learning rate    : 0.03
```

Final result:

```text
Training loss    : 0.4501
Training accuracy: 88.40%
Test loss        : 0.4263
Test accuracy    : 87.40%
```

Example:

```text
examples/mnist_classification.py
```

Notebook:

```text
notebooks/06_mnist.ipynb
```

Documentation:

```text
docs/mnist.md
```

---

## CNN From Scratch

RawGrad also contains a convolutional neural-network implementation.

The CNN pipeline includes:

```text
Input
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
```

Implemented components include:

- 2D convolution
- convolution backward propagation
- max pooling
- max pooling backward propagation
- flattening
- dense layers
- activation functions
- softmax classification

The CNN implementation is validated using both analytical shape checks and numerical gradient comparison.

### CNN Gradient Validation

A numerical filter-gradient comparison produced:

```text
Numerical gradient : 0.04004529323697881
Analytical gradient: 0.04004529323648309
Absolute error     : approximately 4.96e-13
```

The CNN components therefore pass the implemented numerical gradient validation.

### CNN Demonstration

A small CNN experiment is included for demonstration:

```text
examples/cnn_classification.py
```

Notebook:

```text
notebooks/07_cnn.ipynb
```

Documentation:

```text
docs/cnn.md
```

The demonstration uses a deliberately small configuration and should be treated as an implementation validation experiment rather than a competitive MNIST benchmark.

---

## PyTorch Comparison

RawGrad includes a selected numerical comparison against PyTorch.

The comparison checks corresponding numerical behavior for:

- predictions
- MSE loss
- weight gradients
- bias gradients

The comparison uses numerical tolerance checks rather than assuming exact floating-point equality.

Verified result:

```text
RAWGRAD ↔ PYTORCH COMPARISON: PASS
```

Example:

```text
examples/pytorch_comparison.py
```

Notebook:

```text
notebooks/08_pytorch_comparison.ipynb
```

PyTorch is used only for comparison and validation. It is not used to implement RawGrad's core training machinery.

---

## Experiments

The project contains reproducible experiments covering:

- gradient checking
- mini-batch training
- weight initialization
- optimizer comparison
- L2 regularization
- MNIST classification
- CNN validation
- PyTorch numerical comparison

Experiment documentation:

```text
docs/experiments.md
```

Formal experiment report:

```text
reports/experiments.md
```

---

## Test Suite

RawGrad includes automated tests covering the framework's core functionality.

Run:

```powershell
python -m pytest -q
```

Current verified result:

```text
31 passed in 1.26s
```

The test suite covers areas including:

- dense layers
- activations
- losses
- forward propagation
- backward propagation
- optimizers
- gradient checking
- MNIST loading
- regularization
- CNN components
- parameter updates
- framework-level review checks

Python source compilation is also verified with:

```powershell
python -m compileall -q src examples tests
```

---

## Project Structure

```text
RawGrad/
│
├── src/
│   └── rawgrad/
│       ├── activations.py
│       ├── classification_losses.py
│       ├── convolution.py
│       ├── flatten.py
│       ├── layers.py
│       ├── losses.py
│       ├── mnist.py
│       ├── model.py
│       ├── optimizers.py
│       └── pooling.py
│
├── examples/
│   ├── 01_single_neuron.py
│   ├── 02_single_neuron.py
│   ├── 03_dense_layer.py
│   ├── ...
│   ├── 23_l2_regularization.py
│   ├── cnn_classification.py
│   ├── mnist_classification.py
│   └── pytorch_comparison.py
│
├── notebooks/
│   ├── 01_forward_propagation.ipynb
│   ├── 02_backpropagation.ipynb
│   ├── 03_gradient_descent.ipynb
│   ├── 04_optimizers.ipynb
│   ├── 05_regularization.ipynb
│   ├── 06_mnist.ipynb
│   ├── 07_cnn.ipynb
│   └── 08_pytorch_comparison.ipynb
│
├── tests/
│   ├── test_core.py
│   ├── test_gradient_checking.py
│   ├── test_mnist.py
│   ├── test_regularization.py
│   └── test_review.py
│
├── docs/
│   ├── architecture.md
│   ├── mathematics.md
│   ├── backpropagation.md
│   ├── optimizers.md
│   ├── regularization.md
│   ├── gradient_checking.md
│   ├── mnist.md
│   ├── cnn.md
│   └── experiments.md
│
├── reports/
│   ├── experiments.md
│   └── final_report.md
│
├── data/
│   └── mnist/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── .gitattributes
```

---

## Documentation

The documentation is organized by concept:

| Document | Purpose |
|---|---|
| `architecture.md` | Framework architecture and design |
| `mathematics.md` | Mathematical foundations |
| `backpropagation.md` | Backpropagation derivation and implementation |
| `optimizers.md` | Optimization algorithms |
| `regularization.md` | L2 regularization |
| `gradient_checking.md` | Numerical gradient validation |
| `mnist.md` | MNIST pipeline |
| `cnn.md` | CNN implementation |
| `experiments.md` | Experimental results and reproducibility |

The formal project reports are located in:

```text
reports/
```

---

## Design Philosophy

RawGrad follows a few principles:

### 1. Transparency over abstraction

The implementation intentionally exposes the operations that high-level frameworks normally hide.

### 2. Mathematics connected to code

Each major mathematical operation is represented by executable code rather than being treated only as theory.

### 3. Numerical validation

Important gradient calculations are checked numerically wherever practical.

### 4. Small components

Layers, activations, losses, optimizers, and models are kept modular so individual components can be tested independently.

### 5. Honest experimentation

The experiments are intended to demonstrate implementation correctness and learning outcomes rather than present artificially optimized benchmark numbers.

---

## Limitations

RawGrad is an educational and experimental framework rather than a production deep-learning library.

Current limitations include:

- NumPy-based CPU execution
- no GPU backend
- no automatic differentiation engine
- limited layer collection compared with mature frameworks
- limited data-loading utilities
- limited model serialization
- CNN demonstrations use small configurations
- no production-scale distributed training
- no automatic mixed precision
- no computational graph compiler

These limitations are intentional within the scope of the project.

The purpose of RawGrad is to make the mechanics of neural-network training understandable and inspectable.

---

## What This Project Demonstrates

RawGrad demonstrates practical understanding of:

- NumPy-based numerical computing
- neural-network mathematics
- forward propagation
- computational graphs
- backpropagation
- gradient calculation
- gradient descent
- mini-batch training
- optimization
- parameter initialization
- regularization
- numerical gradient checking
- image classification
- convolutional neural networks
- software testing
- experiment design
- reproducibility
- framework-level engineering

It also demonstrates the ability to move from mathematical definitions to working implementations and then validate those implementations experimentally.

---

## Reproducibility

The project records the important experimental configurations and results used during development.

For the main experiments, the repository contains:

- source implementations
- example scripts
- notebooks
- tests
- documentation
- experiment reports
- dataset-loading utilities

The goal is for another developer to inspect the implementation, execute the examples, run the tests, and understand how the reported results were obtained.

---

## Final Status

RawGrad currently includes:

```text
Core neural-network framework       COMPLETE
Forward propagation                 COMPLETE
Backpropagation                     COMPLETE
Gradient descent                    COMPLETE
Mini-batch training                 COMPLETE
Optimizers                          COMPLETE
Regularization                      COMPLETE
Gradient checking                   COMPLETE
MNIST classification                COMPLETE
CNN implementation                  COMPLETE
PyTorch comparison                  COMPLETE
Automated tests                     31 PASSING
Documentation                       COMPLETE
Experiment reports                  COMPLETE
Notebooks                           COMPLETE
CI configuration                    INCLUDED
```

The final verified test status is:

```text
31 passed
```

---

## Future Extensions

Possible future improvements include:

- additional activation functions
- additional optimizers
- batch normalization
- dropout
- more convolutional layers
- data augmentation
- model serialization
- learning-rate schedulers
- GPU acceleration
- automatic differentiation
- broader benchmark comparisons

These are potential extensions rather than claims about the current implementation.

---

## License

RawGrad is released under the license included in this repository.

See:

```text
LICENSE
```

---

## Author

**Sri Varshini S.**

RawGrad was built as a from-scratch exploration of neural-network mathematics, implementation, experimentation, and software engineering using Python and NumPy.