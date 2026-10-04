
import numpy as np
from rawgrad.layers import Dense
from rawgrad.activations import ReLU, Sigmoid
from rawgrad.model import Sequential
from rawgrad.classification_losses import BinaryCrossEntropy
from rawgrad.optimizers import SGD
#CREATE THE DATASET
np.random.seed(42)
X = np.linspace(0.0, 1.0, 201).reshape(-1, 1)
# Class 1 occupies the middle interval.
# Class 0 occupies the two outer intervals.
y = (
    ((X >= 0.25) & (X <= 0.75))
    .astype(float)
)
print("Complete dataset shape:", X.shape)
print("Labels shape:", y.shape)
# shuffle and split data
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]
split_index = int(0.80 * len(X))
X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]
print("\nDATASET SPLIT")
print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))
# BUILD THE MODEL
model = Sequential([
    Dense(1, 8),
    ReLU(),
    Dense(8, 1),
    Sigmoid()
])
loss_function = BinaryCrossEntropy()
optimizer = SGD(learning_rate=0.03)
epochs = 4000
# TRAIN ONLY ON TRAINING DATA
for epoch in range(epochs):
    total_loss = 0.0
    # Shuffle training examples every epoch.
    epoch_indices = np.random.permutation(len(X_train))
    for i in epoch_indices:
        sample = X_train[i]
        target = y_train[i]
        # Forward propagation
        prediction = model.forward(sample)
        # Calculate loss
        loss = loss_function.forward(
            prediction,
            target
        )
        total_loss += loss
        # Calculate loss gradient
        gradient = loss_function.backward()
        # Backpropagation
        model.backward(gradient)
        # Update weights and biases
        optimizer.step(model)
    average_loss = total_loss / len(X_train)
    if (epoch + 1) % 400 == 0:
        print(
            f"Epoch {epoch + 1:4d}"
            f" | Training Loss: {average_loss:.6f}"
        )
# EVALUATION FUNCTION
def evaluate(model, X_data, y_data):
    predictions = []
    actual_values = []
    probabilities = []
    total_loss = 0.0
    for i in range(len(X_data)):
        sample = X_data[i]
        target = y_data[i]
        probability = model.forward(sample)
        probability_value = float(probability[0])
        predicted_class = (
            1 if probability_value >= 0.5 else 0
        )
        actual_class = int(target[0])
        probabilities.append(probability_value)
        predictions.append(predicted_class)
        actual_values.append(actual_class)
        total_loss += loss_function.forward(
            probability,
            target
        )
    predictions = np.array(predictions)
    actual_values = np.array(actual_values)
    # Confusion matrix counts
    true_positive = int(np.sum(
        (actual_values == 1) & (predictions == 1)
    ))
    true_negative = int(np.sum(
        (actual_values == 0) & (predictions == 0)
    ))
    false_positive = int(np.sum(
        (actual_values == 0) & (predictions == 1)
    ))
    false_negative = int(np.sum(
        (actual_values == 1) & (predictions == 0)
    ))
    accuracy = (
        (true_positive + true_negative)
        / len(actual_values)
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    average_loss = total_loss / len(X_data)
    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative
    }
#EVALUATE TRAINING AND TEST DATA
train_results = evaluate(model, X_train, y_train)
test_results = evaluate(model, X_test, y_test)
def print_results(name, results):
    print(f"\n{name}")
    print("-" * 35)
    print(f"Loss      : {results['loss']:.6f}")
    print(f"Accuracy  : {results['accuracy'] * 100:.2f}%")
    print(f"Precision : {results['precision'] * 100:.2f}%")
    print(f"Recall    : {results['recall'] * 100:.2f}%")
    print(f"F1-score  : {results['f1'] * 100:.2f}%")
print_results("TRAINING SET RESULTS", train_results)
print_results("TEST SET RESULTS", test_results)
#Confusion matrix
print("\nTEST CONFUSION MATRIX")
print("Rows = Actual | Columns = Predicted")
print()
print("                 Predicted 0  Predicted 1")
print(
    f"Actual 0             {test_results['tn']:3d}"
    f"           {test_results['fp']:3d}"
)
print(
    f"Actual 1             {test_results['fn']:3d}"
    f"           {test_results['tp']:3d}"
)
# Comparing training and test acc
train_accuracy = train_results["accuracy"]
test_accuracy = test_results["accuracy"]
print("\nGENERALIZATION CHECK")
print("-" * 35)
print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Test accuracy    : {test_accuracy * 100:.2f}%")
accuracy_gap = abs(train_accuracy - test_accuracy)
print(f"Accuracy gap     : {accuracy_gap * 100:.2f} percentage points")
print("\nEvaluation complete.")