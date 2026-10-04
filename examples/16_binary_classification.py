import numpy as np
from rawgrad.layers import Dense
from rawgrad.activations import Sigmoid
from rawgrad.model import Sequential
from rawgrad.classification_losses import BinaryCrossEntropy
from rawgrad.optimizers import SGD
#dataset
X = np.array([
    [0.1],
    [0.2],
    [0.3],
    [0.7],
    [0.8],
    [0.9]
])
y = np.array([
    [0.0],
    [0.0],
    [0.0],
    [1.0],
    [1.0],
    [1.0]
])
#model
model = Sequential([
    Dense(1,1),
    Sigmoid()
])
#loss and optimizer
loss_function = BinaryCrossEntropy()
optimizer = SGD(learning_rate=0.1)
#training
epochs = 1000
for epoch in range(epochs):
    total_loss = 0.0
    for i in range(len(X)):
        sample = X[i]
        target = y[i]
        #fp
        prediction = model.forward(sample)
        #calculate loss
        loss = loss_function.forward(prediction , target)
        total_loss += loss
        #loss gradient
        gradient = loss_function.backward()
        #backpropagation
        model.backward(gradient)
        #update weights and gradients
        optimizer.step(model)
    average_loss = total_loss / len(X)   
    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1:4d}"
            f" | Loss: {average_loss:.6f}"  
        )
#final predictions
print("CLASSIFICATION RESULTS")
correct = 0
for i in range(len(X)):
    sample = X[i]
    target = int(y[i,0])
    probability = model.forward(sample)
    probability_value = float(probability[0])
    predicted_class = (
        1 if probability_value >= 0.5 else 0
    )
    if predicted_class == target:
        correct += 1
    print(
        f"Input:{sample[0]:.1f}"
        f"| Probability : {probability_value:.4f}"
        f"| Predicted : {predicted_class}"
        f"| Actual: {target}"
    )
accuracy = correct / len(X) * 100
print(f"\n Accuracy: {accuracy:.2f}%")

        