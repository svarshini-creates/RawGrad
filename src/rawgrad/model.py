class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        output = inputs

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

        return gradient

    def regularization_loss(self):
        total = 0.0

        for layer in self.layers:
            if hasattr(layer, "regularization_loss"):
                total += layer.regularization_loss()

        return total
