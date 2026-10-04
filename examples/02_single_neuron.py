import numpy as np
#inp
x = np.array([1,2,3,4])
#weight
w = np.array([0.2,0.5,0.1,0.4])
#bias
b = -10.0
#fp
z = np.dot(w,x) + b
print("Weighted Sum:" , z)
#ReLU 
a = np.maximum(0,z)
print("Neuron Output:", a)