import numpy as np
from rawgrad.layers import Dense
#inputs
layer = Dense(4,3)
#inputs
x = np.array([1.0,2.0,3.0,4.0])
#forward pass
output = layer.forward(x)
print("Weights:")
print(layer.weights)
print("\n Biases:")
print(layer.biases)
print("\n Output:")
print(output)
