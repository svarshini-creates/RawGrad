import numpy as np
class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)
    def backward(self, gradient):
        relu_gradient = gradient.copy()
        relu_gradient[self.inputs <= 0] = 0
        return relu_gradient
class Sigmoid:
    def forward(self, inputs):
        # Clip inputs to avoid overflow for extreme values.
        stable_inputs = np.clip(inputs, -500.0, 500.0)
        self.output = 1.0 / (1.0 + np.exp(-stable_inputs))
        return self.output
    def backward(self, gradient):
        sigmoid_gradient = self.output * (1.0 - self.output)
        return gradient * sigmoid_gradient
class Softmax:
    def forward(self, inputs):
        # Subtract the maximum for numerical stability.
        shifted_inputs = inputs - np.max(
            inputs, axis=-1, keepdims=True
        )
        exponentials = np.exp(shifted_inputs)
        self.output = exponentials / np.sum(
            exponentials, axis=-1, keepdims=True
        )
        return self.output
    def backward(self, gradient):
        # Single sample: shape (number_of_classes,).
        if self.output.ndim == 1:
            return self.output * (
                gradient - np.sum(gradient * self.output)
            )
        # Mini-batch: shape (batch_size, number_of_classes).
        return self.output * (
            gradient
            - np.sum(
                gradient * self.output,
                axis=1,
                keepdims=True,
            )
        )