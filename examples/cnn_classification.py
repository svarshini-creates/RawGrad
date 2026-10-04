import numpy as np
from rawgrad.mnist import load_mnist
from rawgrad.convolution import Conv2D
from rawgrad.pooling import MaxPool2D
from rawgrad.flatten import Flatten
from rawgrad.layers import Dense
from rawgrad.activations import ReLU, Softmax
from rawgrad.classification_losses import CategoricalCrossEntropy
from rawgrad.model import Sequential
from rawgrad.optimizers import SGD
def main():
    np.random.seed(42)
    print("RAWGRAD: CNN FROM SCRATCH - MNIST")
    #  Load the MNIST dataset.
    print("\nLoading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist()
    # Shuffle indices so the small subset includes varied digits.
    train_indices = np.random.permutation(len(X_train))[:256]
    test_indices = np.random.permutation(len(X_test))[:100]
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]
    # Convert flattened images into channels-last image format.
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    # Convert integer labels into one-hot vectors.
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    print("Training images:", X_train.shape)
    print("Testing images :", X_test.shape)
    print("Number of classes:", 10)
    #  Build the CNN.
    model = Sequential([
        Conv2D(
            num_filters=2,
            kernel_size=3,
            input_channels=1,
        ),
        ReLU(),
        MaxPool2D(
            pool_size=2,
            stride=2,
        ),
        Flatten(),
        Dense(
            input_size=338,
            output_size=10,
            initialization="xavier",
        ),
        Softmax(),
    ])

    loss_function = CategoricalCrossEntropy()
    optimizer = SGD(learning_rate=0.01)
    #  Train using mini-batches.
    epochs = 3
    batch_size = 8
    print("\nStarting CNN training...")
    for epoch in range(epochs):
        shuffled_indices = np.random.permutation(len(X_train))
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for start in range(0, len(X_train), batch_size):
            batch_indices = shuffled_indices[
                start:start + batch_size
            ]
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            # Forward propagation.
            predictions = model.forward(batch_X)
            loss = loss_function.forward(predictions, batch_y)
            # Record batch statistics before the parameter update.
            total_loss += loss * len(batch_X)
            predicted_classes = np.argmax(predictions, axis=1)
            actual_classes = np.argmax(batch_y, axis=1)
            correct_predictions += np.sum(
                predicted_classes == actual_classes
            )
            total_samples += len(batch_X)
            # Backpropagation and parameter updates.
            gradient = loss_function.backward()
            model.backward(gradient)
            optimizer.step(model)
        average_loss = total_loss / total_samples
        accuracy = 100.0 * correct_predictions / total_samples
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Training loss: {average_loss:.4f} | "
            f"Training accuracy: {accuracy:.2f}%"
        )
    # 4. Evaluate on held-out test images.
    print("\nEvaluating on test images...")
    test_loss_total = 0.0
    test_correct = 0
    test_count = 0
    for start in range(0, len(X_test), batch_size):
        batch_X = X_test[start:start + batch_size]
        batch_y = y_test[start:start + batch_size]
        predictions = model.forward(batch_X)
        loss = loss_function.forward(predictions, batch_y)
        test_loss_total += loss * len(batch_X)
        predicted_classes = np.argmax(predictions, axis=1)
        actual_classes = np.argmax(batch_y, axis=1)
        test_correct += np.sum(predicted_classes == actual_classes)
        test_count += len(batch_X)
    test_loss = test_loss_total / test_count
    test_accuracy = 100.0 * test_correct / test_count
    print("CNN CLASSIFICATION RESULTS")
    print("Training samples :", len(X_train))
    print("Testing samples  :", len(X_test))
    print("Training epochs  :", epochs)
    print("Batch size       :", batch_size)
    print(f"Test loss        : {test_loss:.4f}")
    print(f"Test accuracy    : {test_accuracy:.2f}%")
    # Show sample predictions.
    print("\nSample predictions:")
    sample_count = min(10, len(X_test))
    sample_predictions = model.forward(X_test[:sample_count])
    for index in range(sample_count):
        actual = np.argmax(y_test[index])
        predicted = np.argmax(sample_predictions[index])
        confidence = np.max(sample_predictions[index]) * 100
        print(
            f"Actual: {actual} | "
            f"Predicted: {predicted} | "
            f"Confidence: {confidence:.2f}%"
        )
    #  Verify model output and gradients.
    assert sample_predictions.shape == (sample_count, 10)
    assert np.isfinite(sample_predictions).all()
    assert np.allclose(
        sample_predictions.sum(axis=1),
        1.0,
        atol=1e-6,
    )
    assert np.isfinite(test_loss)
    conv_layer = model.layers[0]
    dense_layer = model.layers[4]
    assert np.isfinite(conv_layer.filters).all()
    assert np.isfinite(dense_layer.weights).all()
    print("\nOutput and parameter checks passed.")
    print("CNN CLASSIFICATION PIPELINE: PASS")
if __name__ == "__main__":
    main()