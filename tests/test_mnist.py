import unittest

import numpy as np

from rawgrad.mnist import load_mnist


class TestMNIST(unittest.TestCase):

    def test_mnist_dataset(self):
        X_train, y_train, X_test, y_test = (
            load_mnist()
        )

        self.assertEqual(
            X_train.shape,
            (60000, 784),
        )

        self.assertEqual(
            y_train.shape,
            (60000,),
        )

        self.assertEqual(
            X_test.shape,
            (10000, 784),
        )

        self.assertEqual(
            y_test.shape,
            (10000,),
        )

        self.assertGreaterEqual(
            X_train.min(),
            0.0,
        )

        self.assertLessEqual(
            X_train.max(),
            1.0,
        )

        self.assertGreaterEqual(
            X_test.min(),
            0.0,
        )

        self.assertLessEqual(
            X_test.max(),
            1.0,
        )

        self.assertTrue(
            np.all(
                (y_train >= 0)
                & (y_train <= 9)
            )
        )

        self.assertTrue(
            np.all(
                (y_test >= 0)
                & (y_test <= 9)
            )
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )