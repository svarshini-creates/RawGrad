
import numpy as np
from rawgrad.mnist import load_mnist
from rawgrad.layers import Dense
from rawgrad.activations import ReLU, Softmax
from rawgrad.classification_losses import CategoricalCrossEntropy
from rawgrad.model import Sequential
from rawgrad.optimizers import SGD
#reproducibility
np.random.seed(42)
# Load MNIST dataset
print("Loading MNIST dataset...")
X_train, y_train, X_test, y_test = load_mnist()
# Use a subset first so training is manageable.
train_indices = np.random.permutation(len(X_train))[:5000]
test_indices = np.random.permutation(len(X_test))[:1000]
X_train = X_train[train_indices]
y_train = y_train[train_indices]
X_test = X_test[test_indices]
y_test = y_test[test_indices]
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Input features:", X_train.shape[1])
print("Number of classes:", len(np.unique(y_train)))
#  Build the neural network
model = Sequential([
    Dense(784, 64, initialization="he"),
    ReLU(),
    Dense(64, 10, initialization="xavier"),
    Softmax(),
])
loss_function = CategoricalCrossEntropy()
optimizer = SGD(learning_rate=0.03)
#train config
epochs = 5
batch_size = 64
# Train the network using mini-batches
print("\nStarting MNIST training...")
for epoch in range(epochs):
    # Shuffle the training examples each epoch.
    shuffled_indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[shuffled_indices]
    y_shuffled = y_train[shuffled_indices]
    total_loss = 0.0
    correct_predictions = 0
    samples_seen = 0
    for start in range(0, len(X_train), batch_size):
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        # Convert integer labels into one-hot vectors.
        y_batch_one_hot = np.eye(10)[y_batch]
        # Forward propagation.
        predictions = model.forward(X_batch)
        # Calculate categorical cross-entropy
        loss = loss_function.forward(
            predictions,
            y_batch_one_hot,
        )
        # Backpropagation.
        gradient = loss_function.backward()
        model.backward(gradient)
        # Update weights and biases.
        optimizer.step(model)
        # Track training statistics.
        batch_samples = len(X_batch)
        total_loss += loss * batch_samples
        predicted_classes = np.argmax(
            predictions,
            axis=1,
        )
        correct_predictions += np.sum(
            predicted_classes == y_batch
        )
        samples_seen += batch_samples
    epoch_loss = total_loss / samples_seen
    epoch_accuracy = (
        correct_predictions / samples_seen
    ) * 100
    print(
        f"Epoch {epoch + 1:02d}/{epochs} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )
#evaluate
print("\nEvaluating on test data...")
test_predictions = model.forward(X_test)
y_test_one_hot = np.eye(10)[y_test]
test_loss = loss_function.forward(
    test_predictions,
    y_test_one_hot,
)
predicted_classes = np.argmax(
    test_predictions,
    axis=1,
)
test_accuracy = np.mean(
    predicted_classes == y_test
) * 100
#display eval
print("RAWGRAD: MNIST CLASSIFICATION RESULTS")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Training epochs  : {epochs}")
print(f"Batch size       : {batch_size}")
print(f"Test loss        : {test_loss:.4f}")
print(f"Test accuracy    : {test_accuracy:.2f}%")
print("\nSample predictions:")
for index in range(10):
    actual = y_test[index]
    predicted = predicted_classes[index]
    confidence = test_predictions[index, predicted] * 100
    print(
        f"Actual: {actual} | "
        f"Predicted: {predicted} | "
        f"Confidence: {confidence:.2f}%"
    )
#verify output
assert test_predictions.shape == (1000, 10)
assert np.isfinite(test_predictions).all()
assert np.allclose(
    test_predictions.sum(axis=1),
    1.0,
)
print("\nAll output checks passed.")
print("MNIST CLASSIFICATION PIPELINE: PASS")
