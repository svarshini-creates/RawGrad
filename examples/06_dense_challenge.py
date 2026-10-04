import numpy as np
from rawgrad.layers import Dense
from rawgrad.activations import ReLU
#initialize inputs
x = np.array([2.5, -1.2, 0.4, 3.1 , -0.8])
#instantiate
dense = Dense(5,4)
relu = ReLU()
#forward pass through dense layer
z = dense.forward(x)
#forward pass through relu layer
output = relu.forward(z)
#print shapes to verify
print("Input shapes:",x.shape)
print("Weight shape:", dense.weights.shape)
print("Bias shape:", dense.biases.shape)
print("Output shape:",output.shape )
#intermediate and final activation
print("\n Before ReLU:")
print(z)
print("\n After ReLU:")
print(output)


 