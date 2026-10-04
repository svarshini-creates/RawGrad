# RawGrad — Neural Networks from Scratch

RawGrad is an educational neural-network framework built from scratch using Python and NumPy. It explores how neural networks perform forward propagation, calculate loss, propagate gradients backward, and update parameters during training.

The goal is to understand the mathematics and implementation behind neural networks rather than relying entirely on high-level deep-learning frameworks.

## Features

- Dense layers supporting single samples and batches
- ReLU, Sigmoid, and Softmax activation functions
- Forward propagation and backpropagation
- Loss functions for regression and classification
- SGD and Momentum optimizers
- Standard, Xavier, He, and zero weight initialization
- Mini-batch training
- Numerical gradient checking
- Convolution, max pooling, and flattening layers
- MNIST handwritten-digit classification
- RawGrad versus PyTorch dense-layer comparison
- Automated unit tests for core components

## Project Structure

```text
RawGrad/
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
├── examples/
├── tests/
│   └── test_core.py
├── data/
├── docs/
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements

- Python 3.13 (the development environment used for this project)
- NumPy
- Internet access for the initial MNIST dataset download

PyTorch is optional and is only needed for the comparison example.

## Installation

Activate your virtual environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependency:

```powershell
python -m pip install -r requirements.txt
```

Set the source directory for Python imports:

```powershell
$env:PYTHONPATH = "src"
```

Run commands from the project root directory.

## Running Examples

Run individual examples to explore the implementation:

```powershell
python examples\01_single_neuron.py
python examples\15_train_linear.py
python examples\16_binary_classification.py
python examples\19_gradient_checking.py
python examples\20_mini_batch_training.py
python examples\21_weight_initialization.py
python examples\22_optimizer_comparison.py
python examples\mnist_classification.py
python examples\cnn_classification.py
```

The MNIST example downloads the dataset on its first run if it is not already available locally.

## Running Tests

Run the automated tests from the project root:

```powershell
python -m unittest discover -s tests -v
```

The tests cover dense layers, activations, categorical cross-entropy, convolution, max pooling, and flattening.

## Experimental Results

### Gradient Checking

The gradient-checking experiment compares analytical gradients with numerical approximations.

- Parameters checked: 10
- Maximum absolute error: approximately `1.016e-11`
- Maximum relative error: approximately `5.634e-09`
- Tolerance: `1e-05`

These results indicate close agreement for the tested parameters and configuration.

### MNIST Classification

A dense neural network was trained on a subset of MNIST.

- Training samples: 5,000
- Test samples: 1,000
- Epochs: 5
- Batch size: 64
- Training accuracy: 88.40%
- Test accuracy: 87.40%

These figures describe this particular experiment, not a benchmark for all MNIST models.

### CNN Classification

A small convolutional neural network was tested on a limited MNIST subset.

- Training samples: 256
- Test samples: 100
- Epochs: 3
- Training accuracy: 34.38%
- Test accuracy: 30.00%

This experiment demonstrates the CNN training pipeline. It is not intended to represent competitive image-classification performance.

### PyTorch Comparison

A dense-layer comparison produced matching predictions, loss, weight gradients, and bias gradients between RawGrad and PyTorch for the tested configuration.

### Optimizer and Initialization Experiments

The project includes experiments comparing SGD with Momentum and different weight-initialization methods. Results depend on the dataset, architecture, hyperparameters, and random initialization. No optimizer or initialization method is universally best.

## Learning Roadmap

The project follows these original 16 milestones:

1. Single neuron
2. Multiple neurons
3. Dense layer
4. Activation functions
5. Loss functions
6. Forward propagation
7. Backpropagation
8. Gradient descent
9. Mini-batch training
10. Optimizers
11. Regularization
12. Gradient checking
13. MNIST classification
14. CNN from scratch
15. PyTorch comparison
16. Final RawGrad portfolio project

## Limitations

RawGrad is an educational implementation intended to demonstrate neural-network fundamentals.

- It is not a replacement for production deep-learning frameworks.
- The CNN experiment uses a small dataset subset and limited training.
- Automated test coverage focuses on core components rather than every possible input.
- Performance and numerical behavior may differ from optimized frameworks.

## Future Improvements

Possible future work includes expanded test coverage, additional numerical checks, improved training utilities, and more experiments. Extensions should be evaluated after the current project is finalized.

## License

No license has been selected yet. Until a license is added, reuse and redistribution should not be assumed to be permitted.
