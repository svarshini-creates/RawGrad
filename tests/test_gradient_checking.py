import unittest

import numpy as np

from rawgrad.layers import Dense


class TestGradientChecking(unittest.TestCase):

    def test_dense_weight_gradient_matches_numerical_gradient(self):
        np.random.seed(42)

        layer = Dense(
            2,
            2,
        )

        inputs = np.array([
            [0.3, -0.7],
        ])

        upstream_gradient = np.array([
            [0.4, -0.2],
        ])

        layer.forward(inputs)
        layer.backward(upstream_gradient)

        analytical = layer.weights_gradient.copy()

        numerical = np.zeros_like(
            layer.weights
        )

        epsilon = 1e-5

        for i in range(
            layer.weights.shape[0]
        ):
            for j in range(
                layer.weights.shape[1]
            ):
                original = layer.weights[i, j]

                layer.weights[i, j] = (
                    original + epsilon
                )

                plus_output = layer.forward(
                    inputs
                )

                plus_value = np.sum(
                    plus_output
                    * upstream_gradient
                )

                layer.weights[i, j] = (
                    original - epsilon
                )

                minus_output = layer.forward(
                    inputs
                )

                minus_value = np.sum(
                    minus_output
                    * upstream_gradient
                )

                layer.weights[i, j] = original

                numerical[i, j] = (
                    plus_value - minus_value
                ) / (2 * epsilon)

        relative_error = np.max(
            np.abs(
                analytical - numerical
            )
        ) / max(
            np.max(np.abs(analytical)),
            np.max(np.abs(numerical)),
            1e-12,
        )

        self.assertLess(
            relative_error,
            1e-5,
        )

    def test_l2_gradient_matches_numerical_gradient(self):
        np.random.seed(42)

        layer = Dense(
            2,
            1,
            l2_lambda=0.1,
        )

        inputs = np.array([
            [0.5, -0.3],
        ])

        upstream_gradient = np.array([
            [0.7],
        ])

        layer.forward(inputs)
        layer.backward(upstream_gradient)

        analytical = layer.weights_gradient.copy()

        numerical = np.zeros_like(
            layer.weights
        )

        epsilon = 1e-5

        for i in range(
            layer.weights.shape[0]
        ):
            for j in range(
                layer.weights.shape[1]
            ):
                original = layer.weights[i, j]

                layer.weights[i, j] = (
                    original + epsilon
                )

                plus_output = layer.forward(
                    inputs
                )

                plus_objective = (
                    np.sum(
                        plus_output
                        * upstream_gradient
                    )
                    + layer.regularization_loss()
                )

                layer.weights[i, j] = (
                    original - epsilon
                )

                minus_output = layer.forward(
                    inputs
                )

                minus_objective = (
                    np.sum(
                        minus_output
                        * upstream_gradient
                    )
                    + layer.regularization_loss()
                )

                layer.weights[i, j] = original

                numerical[i, j] = (
                    plus_objective
                    - minus_objective
                ) / (2 * epsilon)

        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=1e-5,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )