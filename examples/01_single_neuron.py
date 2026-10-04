import numpy as np
# 3 inp
x = np.array([2.0,3.0,4.0])
#weight
w = np.array([0.5,0.2,0.8])
#bias
b = 1.0
#fp
z = np.dot(w,x)+b
print("Weighted Sum:", z)
#ReLU activation
a = np.maximum(0,z)
print("Neuron output:",a)

