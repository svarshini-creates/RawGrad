import numpy as np

from rawgrad.layers import Dense
from rawgrad.activations import ReLU
from rawgrad.model import Sequential
from rawgrad.losses import MeanSquaredError

#create neural network
model = Sequential([
    Dense(3,4),
    ReLU(),
    Dense(4,2)
])
#loss function 
loss_function = MeanSquaredError()
#input
x = np.array([
    1.0,
    2.0,
    3.0
])
#target
target = np.array([
    1.0,
    0.0
])
#fp
prediction = model.forward(x)
print("FORWARD PASS")
print("\n Input:")
print(x)
print("\n Prediction:")
print(prediction)
print("\n Target: ")
print(target)
#calculate loss 
loss = loss_function.forward(
    prediction,
    target
)
print("/n Loss:")
print(loss)
#calculate loss gradient
loss_gradient = loss_function.backward()
print("\n Loss Gradient:")
print(loss_gradient)
#backward pass
input_gradient = model.backward(loss_gradient)
print("BACKWARD PASS")
print("\n Gradient reaching input:")
print(input_gradient)
#input gradients
print("PARAMETER GRADIENTS")
for i , layer in enumerate(model.layers):
    if hasattr(layer , "weights_gradient"):
        print(f"\n Layer {i+1}: {layer.__class__.__name__}")
        print("Weights gradient:")  
        print(layer.weights_gradient)
        print("\n Bias gradient:")
        print(layer.biases_gradient)
        
    