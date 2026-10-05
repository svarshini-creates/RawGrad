import numpy as np
from rawgrad.layers import Dense
from rawgrad.activations import ReLU, Sigmoid
from rawgrad.model import Sequential
from rawgrad.classification_losses import BinaryCrossEntropy
from rawgrad.optimizers import SGD
#dataset
X = np.linspace(0.0,1.0,21).reshape(-1,1)

y = np.array([
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [0.0]
])
#build neural network
model = Sequential([
    Dense(1,8),
    ReLU(),
    Dense(8,1),
    Sigmoid()
])
#loss and optimizer
loss_function = BinaryCrossEntropy()
optimizer = SGD(learning_rate = 0.03)
epochs = 4000
#training
for epoch in range(epochs):
    total_loss = 0.0
    for i in range(len(X)):
        sample = X[i]
        target = y[i]
        #fp
        prediction = model.forward(sample)
        #binary cross-entropy
        loss = loss_function.forward(prediction , target)
        total_loss += loss
        #loss gradient
        gradient = loss_function.backward()
        #backpropagation
        model.backward(gradient)
        #trainable parameters
        optimizer.step(model)
    average_loss = total_loss / len(X)
    if (epoch + 1) % 400 == 0:
        print(
            f"Epoch {epoch + 1:4d}"
            f"| Loss:{average_loss:.6f}"
        )
#evaluate model
print("MULTILAYER CLASSIFICATION RESULTS")
correct = 0
for i in range(len(X)):
    sample = X[i]
    actual = int(y[i,0])
    probability = model.forward(sample)
    probability_value = float(probability[0])
    predicted = (
        1 if probability_value >= 0.5 else 0
    )
    if predicted == actual:
        correct += 1
    print(
        f"Input :{sample[0]:.2f}"
        f"| Probability :{probability_value:.4f}"
        f"| Predicted : {predicted}"
        f"| Actual : {actual}"
    )

accuracy = correct / len(X) * 100
print(f"\n Accuracy :{accuracy:.2f}%")
print(f"Correct predictions:{correct} / {len(X)}")
