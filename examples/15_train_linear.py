import numpy as np
from rawgrad.layers import Dense
from rawgrad.model import Sequential
from rawgrad.losses import MeanSquaredError
from rawgrad.optimizers import SGD
#Dataset
x = np.array([[1.0],[2.0],[3.0],[4.0]])
y = np.array([[2.0],[4.0],[6.0],[8.0]])
#model
model = Sequential([Dense(1,1)])
#loss
loss_function = MeanSquaredError()
#optimizer
optimizer = SGD(learning_rate=0.01)
#training
epochs = 100
for epoch in range(epochs):
    total_loss = 0.0
    for i in range(len(x)):
        X = x[i]
        target = y[i]
        #forward pass
        prediction = model.forward(X)
        #calculate loss
        loss = loss_function.forward(prediction , target)
        total_loss += loss
        #calculate loss gradient
        gradient = loss_function.backward()
        #backpropagation
        model.backward(gradient)
        #update weights
        optimizer.step(model)
    average_loss = total_loss / len(x)
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}"
            f"|Loss : {average_loss:.6f}"
        )
#final predictions
print("FINAL PREDICTIONS")
for X in x:
    prediction = model.forward(X)
    print(
        f"X = {X[0]: .1f}"
        f"-> prediction = {prediction[0]:.4f}"
    )
    