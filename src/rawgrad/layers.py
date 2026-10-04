import numpy as np
class Dense:
    def __init__(
        self,
        input_size,
        output_size,
        initialization="standard",
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.initialization = initialization
        # Initialize weights using the selected method
        if initialization == "standard":
            self.weights = np.random.randn(
                input_size,
                output_size,
            )
        elif initialization == "xavier":
            standard_deviation = np.sqrt(
                2.0 / (input_size + output_size)
            )
            self.weights = (
                np.random.randn(input_size, output_size)
                * standard_deviation
            )
        elif initialization == "he":
            standard_deviation = np.sqrt(
                2.0 / input_size
            )
            self.weights = (
                np.random.randn(input_size, output_size)
                * standard_deviation
            )
        elif initialization == "zeros":
            self.weights = np.zeros(
                (input_size, output_size)
            )
        else:
            raise ValueError(
                f"Unknown initialization: {initialization}"
            )
        # Biases remain initialized to zero
        self.biases = np.zeros(output_size)
    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(
            inputs,
            self.weights,
        ) + self.biases
    def backward(self, gradient):
        # Case 1: Single input sample
        if self.inputs.ndim == 1:
            self.weights_gradient = np.outer(
                self.inputs,
                gradient,
            )
            self.biases_gradient = gradient.copy()
        # Case 2: Mini-batch of input samples
        else:
            self.weights_gradient = np.dot(
                self.inputs.T,
                gradient,
            )
            self.biases_gradient = np.sum(
                gradient,
                axis=0,
            )
        # Gradient with respect to the inputs
        input_gradient = np.dot(
            gradient,
            self.weights.T,
        )
        return input_gradient
    def parameters(self):
        return [
            self.weights,
            self.biases,
        ]
    def gradient(self):
        return [
            self.weights_gradient,
            self.biases_gradient,
        ]