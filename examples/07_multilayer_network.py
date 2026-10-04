import numpy as np
from rawgrad.layers import Dense
from rawgrad.activations import ReLU
#create layers
dense1 = Dense(5,4)
relu1 = ReLU()
dense2 = Dense(4,3)
relu2 = ReLU()
#input
x = np.array([
    1.0,
    2.0,
    3.0,
    4.0,
    5.0
])
#fp layer1
z1 = dense1.forward(x)
print("After Dense 1:")
a1 = relu1.forward(z1)
print("\nAfter ReLU 1:")
print(a1)
#fp layer2
z2 = dense2.forward(a1)
print("After Dense 2:")
print(z2)
a2 = relu2.forward(z2)
print("\n Final output:")
print(a2)
