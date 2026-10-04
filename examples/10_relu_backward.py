import numpy as np
from rawgrad.activations import ReLU
x = np.array([
      2.5,
      -3.0,
      1.2,
      -0.7
])
#create relu
relu = ReLU()
#fp
output = relu.forward(x)
print("Input:")
print(x)
#bp
gradient = np.array([
    0.4,
    0.8,
    0.2,
    0.6
])
backward_gradient = relu.backward(gradient)
print("\nIncoming gradient:")
print(gradient)
print("\nGradient after ReLU backward:")
print(backward_gradient)