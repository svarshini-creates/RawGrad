import numpy as np
from rawgrad.layers import Dense
from rawgrad.activations import ReLU
from rawgrad.model import Sequential
#create model
model = Sequential([Dense(3,4),
                    ReLU(),
                    Dense(4,2),
                    ReLU()
                    ])
#input
x = np.array([1.0,2.0,3.0])
#fp
output = model.forward(x)
print("Input:")
print(x)
print("\n Forward output:")
print(output)
print("\n Output shape:")
print(output.shape)
#inc grad
gradient = np.array([1.0,1.0])
#bp
input_gradient = model.backward(gradient)
print("\n Incoming gradient")
print(gradient)
print("\n Input gradient")
print(input_gradient)
print("\n Input gradient shape:")
print(input_gradient.shape)