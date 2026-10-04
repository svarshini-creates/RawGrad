
import numpy as np
from rawgrad.layers import Dense
from rawgrad.activations import ReLU, Sigmoid
from rawgrad.classification_losses import BinaryCrossEntropy
from rawgrad.model import Sequential
from rawgrad.optimizers import SGD
#config
np.random.seed(7)
EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.03
#create a binary classification report
X = np.linspace(-1.0, 1.0, 201).reshape(-1, 1)
y = (X >= 0.0).astype(float)
# Shuffle before splitting
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]
split_index = int(0.8 * len(X))
X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]
print("RAWGRAD: MINI-BATCH GRADIENT DESCENT")
print("Dataset shape :", X.shape)
print("Training data :", X_train.shape)
print("Testing data  :", X_test.shape)
print("Batch size    :", BATCH_SIZE)
#build NN
model = Sequential([
    Dense(1, 8),
    ReLU(),
    Dense(8, 1),
    Sigmoid(),
])
loss_function = BinaryCrossEntropy()
optimizer = SGD(learning_rate=LEARNING_RATE)
#train 
for epoch in range(EPOCHS):
    # Shuffle training samples every epoch
    shuffled_indices = np.random.permutation(
        len(X_train)
    )
    X_shuffled = X_train[shuffled_indices]
    y_shuffled = y_train[shuffled_indices]
    batch_losses = []
    for start in range(0, len(X_train), BATCH_SIZE):
        end = start + BATCH_SIZE
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        # Forward propagation 
        predictions = model.forward(X_batch)
        # Calculate the batch loss
        loss = loss_function.forward(
            predictions,
            y_batch,
        )
        batch_losses.append(loss)
        # Backpropagation
        gradient = loss_function.backward()
        model.backward(gradient)
        # One update per batch
        optimizer.step(model)
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | "
            f"Average batch loss: "
            f"{np.mean(batch_losses):.6f}"
        )
#evaluate model
train_predictions = model.forward(X_train)
test_predictions = model.forward(X_test)
train_loss = loss_function.forward(
    train_predictions,
    y_train,
)
test_loss = loss_function.forward(
    test_predictions,
    y_test,
)
train_classes = (train_predictions >= 0.5).astype(int)
test_classes = (test_predictions >= 0.5).astype(int)
train_accuracy = np.mean(
    train_classes == y_train
) * 100
test_accuracy = np.mean(
    test_classes == y_test
) * 100
print("\nTRAINING RESULTS")
print(f"Training loss    : {train_loss:.6f}")
print(f"Training accuracy: {train_accuracy:.2f}%")
print("\nTEST RESULTS")
print(f"Testing loss     : {test_loss:.6f}")
print(f"Testing accuracy : {test_accuracy:.2f}%")
#check
print("\nBATCH SHAPE CHECK")
print("Input batch shape :", X_test[:8].shape)
print(
    "Output batch shape:",
    model.forward(X_test[:8]).shape,
)
print("\nMini-batch training complete.")