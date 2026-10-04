import numpy as np
from rawgrad.activations import ReLU
from rawgrad.layers import Dense
from rawgrad.model import Sequential
#def arch
model = Sequential([
    Dense(6,8),
    ReLU(),
    Dense(8,5),
    ReLU(),
    Dense(5,2),
    ReLU()
])
#inp
x = np.array([
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0
])
#fp
output = model.forward(x)
#printing shape and final output
print("Input shape:")
print(x.shape)
print("\n Final output:")
print(output)
print("\n Final output shape:")
print(output.shape)