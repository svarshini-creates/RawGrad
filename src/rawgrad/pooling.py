import numpy as np
class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        if pool_size <= 0:
            raise ValueError("pool_size must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        self.pool_size = pool_size
        self.stride = stride
    def forward(self, inputs):
        # Input shape: (batch, height, width, channels)
        if inputs.ndim != 4:
            raise ValueError(
                "Input must have shape "
                "(batch, height, width, channels)."
            )
        batch_size, height, width, channels = inputs.shape
        pool = self.pool_size
        stride = self.stride
        if height < pool or width < pool:
            raise ValueError(
                "Pooling window cannot exceed input dimensions."
            )
        output_height = (height - pool) // stride + 1
        output_width = (width - pool) // stride + 1
        output = np.zeros(
            (batch_size, output_height, output_width, channels),
            dtype=inputs.dtype,
        )
        self.inputs = inputs
        self.max_positions = np.zeros(
            (batch_size, output_height, output_width, channels, 2),
            dtype=int,
        )
        for row in range(output_height):
            for column in range(output_width):
                row_start = row * stride
                column_start = column * stride
                patch = inputs[
                    :,
                    row_start:row_start + pool,
                    column_start:column_start + pool,
                    :,
                ]
                flattened = patch.reshape(
                    batch_size, pool * pool, channels
                )
                max_indices = np.argmax(flattened, axis=1)
                output[:, row, column, :] = np.max(
                    flattened, axis=1
                )
                self.max_positions[:, row, column, :, 0] = (
                    row_start + max_indices // pool
                )
                self.max_positions[:, row, column, :, 1] = (
                    column_start + max_indices % pool
                )
        return output
    def backward(self, gradient):
        batch_size, height, width, channels = self.inputs.shape
        output_height, output_width = gradient.shape[1:3]
        expected_shape = (
            batch_size,
            output_height,
            output_width,
            channels,
        )
        if gradient.shape != expected_shape:
            raise ValueError(
                f"Expected gradient shape {expected_shape}, "
                f"received {gradient.shape}."
            )
        input_gradient = np.zeros(
            self.inputs.shape,
            dtype=np.result_type(
            self.inputs.dtype,
            gradient.dtype,
            np.float64,
            ),
        )
        for row in range(output_height):
            for column in range(output_width):
                for batch in range(batch_size):
                    for channel in range(channels):
                        max_row, max_column = (
                            self.max_positions[
                                batch, row, column, channel
                            ]
                        )
                        input_gradient[
                            batch, max_row, max_column, channel
                        ] += gradient[
                            batch, row, column, channel
                        ]

        return input_gradient