import numpy as np
import torch
import torch.nn as nn
from rawgrad.layers import Dense
# Make the experiment reproducible.
np.random.seed(42)
torch.manual_seed(42)
# Create one small dataset.
X = np.array(
    [
        [0.2, 0.5],
        [0.7, 0.1],
        [0.4, 0.9],
        [0.8, 0.6],
    ],
    dtype=np.float32,
)
targets = np.array(
    [
        [0.3],
        [0.8],
        [0.5],
        [0.9],
    ],
    dtype=np.float32,
)
# RawGrad implementation
rawgrad_layer = Dense(2, 1, initialization="xavier")
# Copy identical parameters into both implementations.
initial_weights = np.array([[0.2], [-0.4]], dtype=np.float32)
initial_biases = np.array([0.1], dtype=np.float32)
rawgrad_layer.weights = initial_weights.copy()
rawgrad_layer.biases = initial_biases.copy()
rawgrad_predictions = rawgrad_layer.forward(X)
rawgrad_loss = np.mean((rawgrad_predictions - targets) ** 2)
# Gradient of mean squared error.
rawgrad_output_gradient = (
    2.0 * (rawgrad_predictions - targets) / targets.size
)
rawgrad_layer.backward(rawgrad_output_gradient)
# PyTorch implementation
torch_layer = nn.Linear(2, 1)
torch_layer.weight.data = torch.tensor(
    initial_weights.T.copy()
)
torch_layer.bias.data = torch.tensor(
    initial_biases.copy()
)
torch_X = torch.tensor(X)
torch_targets = torch.tensor(targets)
torch_predictions = torch_layer(torch_X)
torch_loss = nn.MSELoss()(torch_predictions, torch_targets)
torch_loss.backward()
#  Compare results
torch_predictions_np = torch_predictions.detach().numpy()
torch_weights_gradient = torch_layer.weight.grad.numpy().T
torch_biases_gradient = torch_layer.bias.grad.numpy()
print("RAWGRAD VS PYTORCH: DENSE LAYER COMPARISON")
print("\nFORWARD PASS")
print("RawGrad predictions:\n", rawgrad_predictions)
print("PyTorch predictions:\n", torch_predictions_np)
print("RawGrad loss:", rawgrad_loss)
print("PyTorch loss:", torch_loss.item())
print("\nBACKWARD PASS")
print("RawGrad weight gradients:\n", rawgrad_layer.weights_gradient)
print("PyTorch weight gradients:\n", torch_weights_gradient)
print("RawGrad bias gradients:", rawgrad_layer.biases_gradient)
print("PyTorch bias gradients:", torch_biases_gradient)
prediction_match = np.allclose(
    rawgrad_predictions, torch_predictions_np, atol=1e-6
)
loss_match = np.isclose(
    rawgrad_loss, torch_loss.item(), atol=1e-6
)
weight_gradient_match = np.allclose(
    rawgrad_layer.weights_gradient,
    torch_weights_gradient,
    atol=1e-6,
)
bias_gradient_match = np.allclose(
    rawgrad_layer.biases_gradient,
    torch_biases_gradient,
    atol=1e-6,
)
print("\nVALIDATION")
print("Predictions match:", prediction_match)
print("Loss matches:", loss_match)
print("Weight gradients match:", weight_gradient_match)
print("Bias gradients match:", bias_gradient_match)
assert prediction_match
assert loss_match
assert weight_gradient_match
assert bias_gradient_match
print("\nRESULT: PASS")
print("RawGrad and PyTorch agree on this Dense layer experiment.")
