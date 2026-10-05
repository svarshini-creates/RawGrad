import unittest
import numpy as np
from rawgrad.layers import Dense
from rawgrad.model import Sequential
class TestL2Regularization(unittest.TestCase):
    def test_regularization_disabled_by_default(self):
        layer = Dense(2, 1)
        layer.weights[:] = [[2.0], [-3.0]]

        self.assertEqual(
            layer.regularization_loss(),
            0.0,
        )
    def test_regularization_penalty(self):
        layer = Dense(
            2,
            1,
            l2_lambda=0.1,
        )
        layer.weights[:] = [[2.0], [-3.0]]
        self.assertAlmostEqual(
            layer.regularization_loss(),
            0.65,
        )
    def test_regularization_gradient(self):
        layer = Dense(
            2,
            1,
            l2_lambda=0.1,
        )
        layer.weights[:] = [[2.0], [-3.0]]
        inputs = np.array([[1.0, 1.0]])
        layer.forward(inputs)
        layer.backward(
            np.array([[1.0]])
        )
        # Data gradient: [1, 1]
        # L2 gradient:   [0.2, -0.3]
        np.testing.assert_allclose(
            layer.weights_gradient,
            [[1.2], [0.7]],
        )
        # Biases are not regularized.
        np.testing.assert_allclose(
            layer.biases_gradient,
            [1.0],
        )
    def test_sequential_sums_layer_penalties(self):
        first = Dense(
            2,
            2,
            l2_lambda=0.1,
        )
        second = Dense(
            2,
            1,
            l2_lambda=0.2,
        )
        first.weights[:] = 1.0
        second.weights[:] = 2.0
        model = Sequential([
            first,
            second,
        ])
        expected = (
            0.5 * 0.1 * 4
            + 0.5 * 0.2 * 8
        )
        self.assertAlmostEqual(
            model.regularization_loss(),
            expected,
        )
    def test_negative_lambda_is_rejected(self):
        with self.assertRaises(ValueError):
            Dense(
                2,
                1,
                l2_lambda=-0.1,
            )
if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )