import unittest
import warnings
import numpy as np
from rawgrad.activations import ReLU, Sigmoid, Softmax
from rawgrad.classification_losses import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
)
from rawgrad.convolution import Conv2D
from rawgrad.flatten import Flatten
from rawgrad.layers import Dense
from rawgrad.model import Sequential
from rawgrad.optimizers import SGD, Momentum
from rawgrad.pooling import MaxPool2D
class TestExactValues(unittest.TestCase):
    def test_dense_forward_values(self):
        layer = Dense(2, 1)
        layer.weights[:] = [[2.0], [3.0]]
        layer.biases[:] = [1.0]
        result = layer.forward(np.array([[4.0, 5.0]]))
        np.testing.assert_allclose(result, [[24.0]])
    def test_relu_values_and_gradient(self):
        layer = ReLU()
        x = np.array([-2.0, 0.0, 3.0])
        result = layer.forward(x)
        gradient = layer.backward(np.ones_like(x))
        np.testing.assert_array_equal(result, [0.0, 0.0, 3.0])
        np.testing.assert_array_equal(gradient, [0.0, 0.0, 1.0])
    def test_softmax_probabilities(self):
        result = Softmax().forward(
            np.array([[2.0, 1.0, 0.0]])
        )
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result.sum(axis=1), [1.0])
    def test_sigmoid_extreme_inputs(self):
        layer = Sigmoid()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = layer.forward(
                np.array([-1000.0, 0.0, 1000.0])
            )
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(result[1], 0.5)
        self.assertFalse(
            any("overflow" in str(w.message).lower() for w in caught)
        )
    def test_max_pool_exact_values(self):
        x = np.array(
            [[
                [[1.0], [3.0], [2.0], [4.0]],
                [[5.0], [6.0], [7.0], [8.0]],
                [[9.0], [2.0], [1.0], [3.0]],
                [[4.0], [5.0], [6.0], [7.0]],
            ]]
        )
        layer = MaxPool2D(pool_size=2, stride=2)
        result = layer.forward(x)
        np.testing.assert_array_equal(
            result.reshape(-1), [6.0, 8.0, 9.0, 7.0]
        )

    def test_flatten_values(self):
        x = np.arange(24).reshape(2, 3, 4, 1)
        layer = Flatten()
        result = layer.forward(x)
        self.assertEqual(result.shape, (2, 12))
        np.testing.assert_array_equal(result[0], x[0].reshape(-1))
class TestGradients(unittest.TestCase):
    def test_dense_weight_gradient_finite_difference(self):
        layer = Dense(2, 1)
        layer.weights[:] = [[0.3], [-0.4]]
        layer.biases[:] = [0.1]
        x = np.array([[0.5, -0.7]])
        upstream = np.array([[1.2]])
        output = layer.forward(x)
        layer.backward(upstream)
        analytical = layer.weights_gradient[0, 0]
        original = layer.weights[0, 0]
        epsilon = 1e-5
        layer.weights[0, 0] = original + epsilon
        plus = np.sum(layer.forward(x) * upstream)
        layer.weights[0, 0] = original - epsilon
        minus = np.sum(layer.forward(x) * upstream)
        layer.weights[0, 0] = original
        numerical = (plus - minus) / (2 * epsilon)
        self.assertAlmostEqual(analytical, numerical, places=6)
    def test_conv_filter_gradient_finite_difference(self):
        np.random.seed(3)
        x = np.random.randn(1, 4, 4, 1)
        layer = Conv2D(1, 2, 1, stride=1)
        upstream = np.ones((1, 3, 3, 1))
        layer.forward(x)
        layer.backward(upstream)
        analytical = layer.filters_gradient[0, 0, 0, 0]
        original = layer.filters[0, 0, 0, 0]
        epsilon = 1e-5
        layer.filters[0, 0, 0, 0] = original + epsilon
        plus = np.sum(layer.forward(x) * upstream)
        layer.filters[0, 0, 0, 0] = original - epsilon
        minus = np.sum(layer.forward(x) * upstream)
        layer.filters[0, 0, 0, 0] = original
        numerical = (plus - minus) / (2 * epsilon)
        self.assertAlmostEqual(analytical, numerical, places=6)
    def test_conv_stride_two_shapes(self):
        x = np.ones((1, 5, 5, 1))
        layer = Conv2D(2, 2, 1, stride=2)
        output = layer.forward(x)
        input_gradient = layer.backward(np.ones_like(output))
        self.assertEqual(output.shape, (1, 2, 2, 2))
        self.assertEqual(input_gradient.shape, x.shape)
    def test_integer_input_keeps_fractional_conv_gradient(self):
        x = np.arange(9).reshape(1, 3, 3, 1)
        layer = Conv2D(1, 2, 1)
        output = layer.forward(x)
        gradient = layer.backward(np.full(output.shape, 0.5))
        self.assertTrue(np.issubdtype(gradient.dtype, np.floating))
        self.assertTrue(np.any(gradient % 1 != 0))
    def test_integer_input_keeps_fractional_pool_gradient(self):
        x = np.array(
            [[
                [[1], [2]],
                [[3], [4]],
            ]]
        )
        layer = MaxPool2D(2, 2)
        output = layer.forward(x)
        gradient = layer.backward(np.array([[[[0.5]]]]))
        self.assertTrue(np.issubdtype(gradient.dtype, np.floating))
        self.assertEqual(gradient[0, 1, 1, 0], 0.5)
class TestLosses(unittest.TestCase):
    def test_binary_cross_entropy_is_finite(self):
        loss = BinaryCrossEntropy()
        predictions = np.array([[0.8], [0.2]])
        targets = np.array([[1.0], [0.0]])
        value = loss.forward(predictions, targets)
        gradient = loss.backward()
        self.assertTrue(np.isfinite(value))
        self.assertTrue(np.isfinite(gradient).all())
    def test_categorical_cross_entropy_is_finite(self):
        loss = CategoricalCrossEntropy()
        predictions = np.array([[0.8, 0.2], [0.1, 0.9]])
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])
        value = loss.forward(predictions, targets)
        gradient = loss.backward()
        self.assertTrue(np.isfinite(value))
        self.assertTrue(np.isfinite(gradient).all())
class TestOptimizers(unittest.TestCase):
    def test_sgd_updates_weight_in_negative_gradient_direction(self):
        layer = Dense(1, 1)
        layer.weights[:] = 2.0
        layer.biases[:] = 0.0
        layer.weights_gradient = np.array([[3.0]])
        layer.biases_gradient = np.array([0.0])
        model = Sequential([layer])
        SGD(learning_rate=0.1).step(model)
        np.testing.assert_allclose(layer.weights, [[1.7]])
    def test_momentum_first_update(self):
        layer = Dense(1, 1)
        layer.weights[:] = 2.0
        layer.biases[:] = 0.0
        layer.weights_gradient = np.array([[3.0]])
        layer.biases_gradient = np.array([0.0])
        model = Sequential([layer])
        Momentum(learning_rate=0.1, momentum=0.9).step(model)
        np.testing.assert_allclose(layer.weights, [[1.7]])
if __name__ == "__main__":
    unittest.main(verbosity=2)