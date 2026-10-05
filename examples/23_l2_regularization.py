import numpy as np
from rawgrad.activations import ReLU
from rawgrad.layers import Dense
from rawgrad.losses import MeanSquaredError
from rawgrad.model import Sequential
from rawgrad.optimizers import SGD
np.random.seed(42)
# Create a small synthetic regression dataset.
X = np.random.randn(200, 2)
true_weights = np.array([
    [2.0],
    [-3.0],
])
y = X @ true_weights + 0.5
# Enable L2 regularization.
model = Sequential([
    Dense(
        2,
        8,
        initialization="he",
        l2_lambda=0.01,
    ),
    ReLU(),
    Dense(
        8,
        1,
        initialization="xavier",
        l2_lambda=0.01,
    ),
])
loss_function = MeanSquaredError()
optimizer = SGD(
    learning_rate=0.01
)
initial_penalty = (
    model.regularization_loss()
)
for epoch in range(300):
    predictions = model.forward(X)
    data_loss = loss_function.forward(
        predictions,
        y,
    )
    gradient = loss_function.backward()
    model.backward(gradient)
    optimizer.step(model)
    if (epoch + 1) % 50 == 0:
        predictions = model.forward(X)
        data_loss = loss_function.forward(
            predictions,
            y,
        )
        penalty = (
            model.regularization_loss()
        )
        print(
            f"Epoch {epoch + 1:3d} | "
            f"Data loss: {data_loss:.6f} | "
            f"L2 penalty: {penalty:.6f} | "
            f"Total objective: "
            f"{data_loss + penalty:.6f}"
        )
print(
    f"\nInitial L2 penalty: "
    f"{initial_penalty:.6f}"
)
print(
    f"Final L2 penalty:   "
    f"{model.regularization_loss():.6f}"
)
print(
    "L2 regularization example completed."
)