class Flatten:
    def forward(self, inputs):
        self.input_shape = inputs.shape
        batch_size = inputs.shape[0]
        return inputs.reshape(batch_size, -1)

    def backward(self, gradient):
        return gradient.reshape(self.input_shape)
