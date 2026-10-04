import numpy as np
from rawgrad.layers import Dense
from rawgrad.activations import ReLU
from rawgrad.model import Sequential
#create the network
model = Sequential([
    Dense(5,4),
    ReLU(),
    Dense(4,3),
    ReLU()
])
#input
x = np.array([1.0,2.0,3.0,4.0,5.0])
#fp
output = model.forward(x)
print("Input:")
print(x)
print("\n Final Output:")
print(output)
print("\n Output shape:")
print(output.shape)