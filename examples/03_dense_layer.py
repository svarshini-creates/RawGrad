import numpy as np
#input
x = np.array([1.0,2.0,3.0,4.0])
#weight
W = np.array([[0.2,0.5,0.1,0.4],
              [0.3,0.7,0.2,0.6],
              [0.9,0.1,0.8,0.5]])

#bias
b = np.array([0.5,0.2,0.1])
#fp
z = np.dot(W,x)+ b
print("Input shape:",x.shape)
print("Weight shape:", W.shape)
print("Bias shape:",b.shape)
print("Output shape:", z.shape)
print("/n Weighted sums:")
print(z)
#ReLU activation
output = np.maximum(0,z)
print("\n Layer output:")
print(output)