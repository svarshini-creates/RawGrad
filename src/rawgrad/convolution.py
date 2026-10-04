import numpy as np
class Conv2D:
    def __init__(
        self,
        num_filters,
        kernel_size,
        input_channels,
        stride=1,
    ):
        if num_filters <= 0:
            raise ValueError("num_filters must be positive.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.stride = stride
        #initialization for convolution filters
        fan_in = kernel_size * kernel_size * input_channels
        standard_deviation = np.sqrt(2.0 / fan_in)
        self.filters = (
            np.random.randn(
                num_filters,
                kernel_size,
                kernel_size,
                input_channels,
            )
            * standard_deviation
        )
        self.biases = np.zeros(num_filters)
    def forward(self, inputs):
        # Expected input: (batch, height, width, channels)
        if inputs.ndim != 4:
            raise ValueError(
                "Input must have shape "
                "(batch, height, width, channels)."
            )
        if inputs.shape[3] != self.input_channels:
            raise ValueError(
                "Input channel count does not match "
                "input_channels."
            )
        self.inputs = inputs
        batch_size, height, width, channels = inputs.shape
        kernel = self.kernel_size
        stride = self.stride
        if height < kernel or width < kernel:
            raise ValueError(
                "Kernel cannot be larger than the input."
            )
        output_height = (height - kernel) // stride + 1
        output_width = (width - kernel) // stride + 1
        output = np.zeros(
            (
                batch_size,
                output_height,
                output_width,
                self.num_filters,
            )
        )
        for row in range(output_height):
            for column in range(output_width):
                row_start = row * stride
                row_end = row_start + kernel
                column_start = column * stride
                column_end = column_start + kernel
                # Extract a patch from every image in the batch.
                patch = inputs[
                    :,
                    row_start:row_end,
                    column_start:column_end,
                    :,
                ]
                # Apply every filter to the patch.
                for filter_index in range(self.num_filters):
                    output[
                        :, row, column, filter_index
                    ] = (
                        np.sum(
                            patch
                            * self.filters[filter_index],
                            axis=(1, 2, 3),
                        )
                        + self.biases[filter_index]
                    )
        return output
    def backward(self, gradient):
        batch_size, height, width, channels = self.inputs.shape
        kernel = self.kernel_size
        stride = self.stride
        expected_shape = (
            batch_size,
            (height - kernel) // stride + 1,
            (width - kernel) // stride + 1,
            self.num_filters,
        )
        if gradient.shape != expected_shape:
            raise ValueError(
                f"Expected gradient shape {expected_shape}, "
                f"received {gradient.shape}."
            )
        output_height = expected_shape[1]
        output_width = expected_shape[2]
        # Gradient accumulators start at zero.
        self.filters_gradient = np.zeros_like(self.filters)
        self.biases_gradient = np.sum(
            gradient,
            axis=(0, 1, 2),
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
                row_start = row * stride
                row_end = row_start + kernel
                column_start = column * stride
                column_end = column_start + kernel
                patch = self.inputs[
                    :,
                    row_start:row_end,
                    column_start:column_end,
                    :,
                ]
                for filter_index in range(self.num_filters):
                    # Gradient flowing through this filter.
                    filter_gradient_values = gradient[
                        :, row, column, filter_index
                    ]
                    # Accumulate gradients for the filter weights.
                    self.filters_gradient[filter_index] += np.sum(
                        patch
                        * filter_gradient_values[
                            :, None, None, None
                        ],
                        axis=0,
                    )
                    # Propagate gradients back to the input images.
                    input_gradient[
                        :,
                        row_start:row_end,
                        column_start:column_end,
                        :,
                    ] += (
                        filter_gradient_values[
                            :, None, None, None
                        ]
                        * self.filters[filter_index][None, :, :, :]
                    )
        return input_gradient
    def parameters(self):
        return [self.filters, self.biases]
    def gradient(self):
        return [self.filters_gradient, self.biases_gradient]