import numpy as np
from rawgrad.layers import Dense
from rawgrad.optimizers  import SGD
#create layer
dense = Dense(3,2)
optimizer = SGD( learning_rate = 0.01)
#input
x = np.array([1.0,2.0,3.0])
#forward
output = dense.forward(x)
print("Initial Weights:")
print(dense.weights)
print("\n Initial biases:")
print(dense.biases)
print("\n Output:")
print(output)
# Example gradient used for the update demonstration
gradient = np.array([
    0.5,
    0.2
])
#backward
dense.backward(gradient)
print("\n Weight gradients:")
print(dense.weights_gradient)
print("\n Bias gradients:")
print(dense.biases_gradient)
#save old parameters
old_weights = dense.weights.copy()
old_biases = dense.biases.copy()
#update tiny wrapper
class SimpleModel:
    def __init__ (self,layer):
        self.layers = [layer]
model = SimpleModel(dense)
optimizer.step(model)
#updated parameters
print("\n Updated Weights:")
print(dense.weights)
print("\n Updated Biases:")
print(dense.biases)
print("\n Weight change:")
print(dense.weights - old_weights)
print("\n Bias change:")
print(dense.biases - old_biases)
