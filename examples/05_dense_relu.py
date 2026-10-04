import numpy as np
from rawgrad.layers import Dense
from rawgrad.activations import ReLU
dense = Dense(4,3)
relu = ReLU()
x = np.array([1.0,2.0,3.0,4.0])
z = dense.forward(x)
print("Dense output:")
print(z)
a = relu.forward(z)
print("\n ReLU output:")
print(a)
