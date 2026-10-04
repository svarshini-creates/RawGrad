
import numpy as np
from rawgrad.layers import Dense
from rawgrad.activations import ReLU, Sigmoid
from rawgrad.model import Sequential
from rawgrad.classification_losses import BinaryCrossEntropy
from rawgrad.optimizers import SGD, Momentum
#create dataset
np.random.seed(42)
X = np.linspace(0.0, 1.0, 201).reshape(-1, 1)
y = ((X >= 0.25) & (X <= 0.75)).astype(float)
indices = np.random.permutation(len(X))
train_indices = indices[:160]
test_indices = indices[160:]
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]
#create model
def create_model():
    return Sequential([
        Dense(1, 8, initialization="he"),
        ReLU(),
        Dense(8, 1, initialization="xavier"),
        Sigmoid(),
    ])
#evaluate model
def evaluate(model, loss_function, X_data, y_data):
    predictions = model.forward(X_data)
    loss = loss_function.forward(predictions, y_data)
    predicted_labels = (predictions >= 0.5).astype(float)
    accuracy = np.mean(predicted_labels == y_data) * 100
    return loss, accuracy
#train
def train_model(optimizer_name, optimizer_class):
    np.random.seed(42)
    model = create_model()
    loss_function = BinaryCrossEntropy()
    optimizer = optimizer_class(
        learning_rate=0.03
    ) if optimizer_name == "SGD" else optimizer_class(
        learning_rate=0.03,
        momentum=0.9
    )
    epochs = 300
    batch_size = 16
    print(f"\nTRAINING WITH {optimizer_name}")
    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(len(X_train))
        for start in range(0, len(X_train), batch_size):
            batch_indices = shuffled_indices[
                start:start + batch_size
            ]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            predictions = model.forward(X_batch)
            loss_function.forward(predictions, y_batch)
            gradient = loss_function.backward()
            model.backward(gradient)
            optimizer.step(model)
        if (epoch + 1) % 50 == 0:
            train_loss, train_accuracy = evaluate(
                model, loss_function, X_train, y_train
            )
            print(
                f"Epoch {epoch + 1:3d} | "
                f"Loss: {train_loss:.6f} | "
                f"Accuracy: {train_accuracy:.2f}%"
            )

    train_loss, train_accuracy = evaluate(
        model, loss_function, X_train, y_train
    )
    test_loss, test_accuracy = evaluate(
        model, loss_function, X_test, y_test
    )
    return {
        "optimizer": optimizer_name,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
#sgd comp
sgd_results = train_model("SGD", SGD)
momentum_results = train_model("Momentum", Momentum)
print("RAWGRAD: OPTIMIZER COMPARISON")
print(
    f"{'Optimizer':<12}"
    f"{'Train Loss':<14}"
    f"{'Train Acc':<14}"
    f"{'Test Loss':<14}"
    f"{'Test Acc':<12}"
)
for result in [sgd_results, momentum_results]:
    print(
        f"{result['optimizer']:<12}"
        f"{result['train_loss']:<14.6f}"
        f"{result['train_accuracy']:<14.2f}"
        f"{result['test_loss']:<14.6f}"
        f"{result['test_accuracy']:<12.2f}"
    )
print("\nOptimizer comparison complete.")