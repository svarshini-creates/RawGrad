import numpy as np
from rawgrad.layers import Dense
#create dense layer
dense = Dense(input_size = 3 ,output_size = 2)
#input
x = np.array([
    1.0,
    2.0,
    3.0
])
#fp
output = dense.forward(x)
print("Input:")
print(x)
print("\n Weights:")
print(dense.weights)
print("\n Biases")
print(dense.biases)
print("\nForward output:")
print(output)
#inc grad
gradient = np.array([0.5,0.2])
#bp
input_gradient = dense.backward(gradient)
print("\n Incoming gradient:")
print(gradient)
print("\n Weights gradient:")
print(dense.weights_gradient)
print("\n Bias gradient:")
print(dense.biases_gradient)
print("\n Input gradient:")
print(input_gradient)
#shape checks
print("\n Shapes:")
print("Input shape:", x.shape)
print("Weights shape:",dense.weights.shape)
print("Bias shape:",dense.biases.shape)
print("Weights gradient shape:",dense.weights_gradient.shape)
print("Bias gradient shape:", dense.biases_gradient.shape)
print("Input gradient shape:",input_gradient.shape)