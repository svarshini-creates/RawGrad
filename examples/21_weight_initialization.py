
import numpy as np
from rawgrad.layers import Dense
from rawgrad.activations import ReLU, Sigmoid
from rawgrad.classification_losses import BinaryCrossEntropy
from rawgrad.model import Sequential
from rawgrad.optimizers import SGD
#config
EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.03
INITIALIZATIONS = [
    "standard",
    "xavier",
    "he",
    "zeros",
]
#create dataset
np.random.seed(7)
X = np.linspace(
    -1.0,
    1.0,
    201,
).reshape(-1, 1)
y = (X >= 0.0).astype(float)
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]
split_index = int(0.8 * len(X))
X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]
#train and evaluate
results = []
for initialization in INITIALIZATIONS:
    # Reset the seed for a controlled experiment
    np.random.seed(42)
    model = Sequential([
        Dense(1, 8, initialization=initialization),
        ReLU(),
        Dense(8, 1, initialization=initialization),
        Sigmoid(),
    ])
    loss_function = BinaryCrossEntropy()
    optimizer = SGD(learning_rate=LEARNING_RATE)
    # Record the initial weight scale
    initial_weight_std = np.std(
        model.layers[0].weights
    )
    print(f"INITIALIZATION: {initialization.upper()}")
    # Mini-batch training
    for epoch in range(EPOCHS):
        shuffled_indices = np.random.permutation(
            len(X_train)
        )
        X_shuffled = X_train[shuffled_indices]
        y_shuffled = y_train[shuffled_indices]
        batch_losses = []
        for start in range(
            0,
            len(X_train),
            BATCH_SIZE,
        ):
            end = start + BATCH_SIZE
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            predictions = model.forward(X_batch)
            loss = loss_function.forward(
                predictions,
                y_batch,
            )
            batch_losses.append(loss)
            gradient = loss_function.backward()
            model.backward(gradient)
            optimizer.step(model)
        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{EPOCHS} | "
                f"Average batch loss: "
                f"{np.mean(batch_losses):.6f}"
            )
    # Evaluate on training data
    train_predictions = model.forward(X_train)
    train_loss = loss_function.forward(
        train_predictions,
        y_train,
    )
    train_classes = (
        train_predictions >= 0.5
    ).astype(int)
    train_accuracy = np.mean(
        train_classes == y_train
    ) * 100
    # Evaluate on test data
    test_predictions = model.forward(X_test)
    test_loss = loss_function.forward(
        test_predictions,
        y_test,
    )
    test_classes = (
        test_predictions >= 0.5
    ).astype(int)
    test_accuracy = np.mean(
        test_classes == y_test
    ) * 100
    results.append({
        "initialization": initialization,
        "initial_std": initial_weight_std,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    })

    print("\nRESULTS")
    print(f"Initial weight std: {initial_weight_std:.6f}")
    print(f"Training loss     : {train_loss:.6f}")
    print(f"Training accuracy : {train_accuracy:.2f}%")
    print(f"Test loss         : {test_loss:.6f}")
    print(f"Test accuracy     : {test_accuracy:.2f}%")
#compare
print("\n")
print("WEIGHT INITIALIZATION COMPARISON")
print(
    f"{'Method':<12}"
    f"{'Initial Std':>13}"
    f"{'Train Loss':>13}"
    f"{'Train Acc':>12}"
    f"{'Test Loss':>13}"
    f"{'Test Acc':>12}"
)
for result in results:
    print(
        f"{result['initialization']:<12}"
        f"{result['initial_std']:>13.6f}"
        f"{result['train_loss']:>13.6f}"
        f"{result['train_accuracy']:>11.2f}%"
        f"{result['test_loss']:>13.6f}"
        f"{result['test_accuracy']:>11.2f}%"
    )
print("\nExperiment complete.")