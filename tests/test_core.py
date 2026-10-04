import unittest
import numpy as np
from rawgrad.activations import ReLU, Softmax
from rawgrad.classification_losses import CategoricalCrossEntropy
from rawgrad.convolution import Conv2D
from rawgrad.flatten import Flatten
from rawgrad.layers import Dense
from rawgrad.pooling import MaxPool2D
class TestDenseLayer(unittest.TestCase):
    def test_forward_output_shape(self):
        layer = Dense(3, 2, initialization="xavier")
        inputs = np.ones((4, 3))
        output = layer.forward(inputs)
        self.assertEqual(output.shape, (4, 2))
        self.assertTrue(np.isfinite(output).all())
    def test_backward_gradient_shapes(self):
        layer = Dense(3, 2, initialization="xavier")
        inputs = np.ones((4, 3))
        output = layer.forward(inputs)
        upstream_gradient = np.ones_like(output)
        input_gradient = layer.backward(upstream_gradient)
        self.assertEqual(input_gradient.shape, inputs.shape)
        self.assertEqual(layer.weights_gradient.shape, layer.weights.shape)
        self.assertEqual(layer.biases_gradient.shape, layer.biases.shape)
class TestActivations(unittest.TestCase):
    def test_relu_forward_and_backward(self):
        layer = ReLU()
        inputs = np.array([-2.0, 0.0, 3.0])
        output = layer.forward(inputs)
        gradient = layer.backward(np.ones_like(output))
        np.testing.assert_array_equal(output, [0.0, 0.0, 3.0])
        np.testing.assert_array_equal(gradient, [0.0, 0.0, 1.0])
    def test_softmax_probabilities_sum_to_one(self):
        layer = Softmax()
        inputs = np.array([[2.0, 1.0, 0.1], [0.2, 0.3, 2.0]])
        probabilities = layer.forward(inputs)
        self.assertEqual(probabilities.shape, inputs.shape)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(inputs.shape[0]),
        )
class TestClassificationLoss(unittest.TestCase):
    def test_categorical_cross_entropy(self):
        probabilities = np.array(
            [[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]
        )
        targets = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )

        loss_function = CategoricalCrossEntropy()
        loss = loss_function.forward(probabilities, targets)
        gradient = loss_function.backward()
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(loss, 0.0)
        self.assertEqual(gradient.shape, targets.shape)
        self.assertTrue(np.isfinite(gradient).all())
class TestCNNComponents(unittest.TestCase):
    def test_convolution_forward_and_backward(self):
        np.random.seed(42)
        inputs = np.random.rand(2, 5, 5, 1)
        layer = Conv2D(
            num_filters=3,
            kernel_size=3,
            input_channels=1,
        )

        output = layer.forward(inputs)
        input_gradient = layer.backward(np.ones_like(output))
        self.assertEqual(output.shape, (2, 3, 3, 3))
        self.assertEqual(input_gradient.shape, inputs.shape)
        self.assertEqual(
            layer.filters_gradient.shape,
            layer.filters.shape,
        )
        self.assertEqual(
            layer.biases_gradient.shape,
            layer.biases.shape,
        )
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(np.isfinite(input_gradient).all())
    def test_max_pool_forward_and_backward(self):
        inputs = np.array(
            [
                [
                    [[1.0], [3.0], [2.0], [4.0]],
                    [[5.0], [6.0], [7.0], [8.0]],
                    [[9.0], [2.0], [1.0], [3.0]],
                    [[4.0], [5.0], [6.0], [7.0]],
                ]
            ]
        )
        layer = MaxPool2D(pool_size=2, stride=2)
        output = layer.forward(inputs)
        input_gradient = layer.backward(np.ones_like(output))
        self.assertEqual(output.shape, (1, 2, 2, 1))
        np.testing.assert_array_equal(
            output.reshape(-1),
            [6.0, 8.0, 9.0, 7.0],
        )
        self.assertEqual(input_gradient.shape, inputs.shape)
        self.assertEqual(input_gradient.sum(), 4.0)
    def test_flatten_forward_and_backward(self):
        inputs = np.arange(72, dtype=float).reshape(2, 3, 3, 4)
        layer = Flatten()
        output = layer.forward(inputs)
        input_gradient = layer.backward(np.ones_like(output))
        self.assertEqual(output.shape, (2, 36))
        self.assertEqual(input_gradient.shape, inputs.shape)
        np.testing.assert_array_equal(output[0], inputs[0].reshape(-1))
        self.assertTrue(np.all(input_gradient == 1.0))
if __name__ == "__main__":
    unittest.main(verbosity=2)
