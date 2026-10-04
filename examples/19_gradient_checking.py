
import numpy as np
from rawgrad.activations import ReLU, Sigmoid
from rawgrad.classification_losses import BinaryCrossEntropy
from rawgrad.layers import Dense
from rawgrad.model import Sequential
# config
RANDOM_SEED = 7
EPSILON = 1e-5
RELATIVE_ERROR_TOLERANCE = 1e-5
#model creation
def build_model():
    return Sequential([
        Dense(1, 3),
        ReLU(),
        Dense(3, 1),
        Sigmoid(),
    ])
# LOSS CALCULATION
def calculate_loss(model, features, targets, loss_function):
    predictions = model.forward(features)
    loss = loss_function.forward(
        predictions,
        targets,
    )
    return float(loss)
def calculate_analytical_gradients(
    model,
    features,
    targets,
    loss_function,
):
    predictions = model.forward(features)
    loss_function.forward(
        predictions,
        targets,
    )
    loss_gradient = loss_function.backward()
    model.backward(loss_gradient)
    analytical_gradients = []
    for layer in model.layers:
        if isinstance(layer, Dense):
            analytical_gradients.append(
                layer.weights_gradient.copy()
            )
            analytical_gradients.append(
                layer.biases_gradient.copy()
            )
    return analytical_gradients
def calculate_numerical_gradient(
    model,
    parameter,
    parameter_index,
    features,
    targets,
    loss_function,
    epsilon=EPSILON,
):
    original_value = parameter[parameter_index]
    try:
        # Evaluate loss at theta + epsilon.
        parameter[parameter_index] = (
            original_value + epsilon
        )
        loss_plus = calculate_loss(
            model,
            features,
            targets,
            loss_function,
        )
        # Evaluate loss at theta - epsilon.
        parameter[parameter_index] = (
            original_value - epsilon
        )
        loss_minus = calculate_loss(
            model,
            features,
            targets,
            loss_function,
        )
        numerical_gradient = (
            loss_plus - loss_minus
        ) / (2.0 * epsilon)
    finally:
        # restore the original parameter.
        parameter[parameter_index] = original_value
    return numerical_gradient
#grad comparison
def relative_error(analytical_gradient, numerical_gradient):
    numerator = abs(
        analytical_gradient - numerical_gradient
    )
    denominator = max(
        abs(analytical_gradient)
        + abs(numerical_gradient),
        1e-12,
    )
    return numerator / denominator
def check_gradients(
    model,
    features,
    targets,
    loss_function,
):
    # Calculate analytical gradients before numerical
    # perturbations overwrite layer forward-pass caches.
    analytical_gradients = calculate_analytical_gradients(
        model,
        features,
        targets,
        loss_function,
    )
    gradient_index = 0
    maximum_relative_error = 0.0
    maximum_absolute_error = 0.0
    total_parameters_checked = 0
    all_gradients_passed = True
    print("\nGRADIENT CHECK RESULTS")
    for layer_number, layer in enumerate(model.layers, start=1):
        if not isinstance(layer, Dense):
            continue
        parameter_names = [
            "Weights",
            "Biases",
        ]
        parameters = layer.parameters()
        for parameter_name, parameter in zip(
            parameter_names,
            parameters,
        ):
            analytical_gradient = analytical_gradients[
                gradient_index
            ]
            gradient_index += 1

            numerical_gradient = np.zeros_like(
                parameter,
                dtype=np.float64,
            )
            # Numerically estimate every element's gradient.
            for parameter_index in np.ndindex(parameter.shape):
                numerical_gradient[parameter_index] = (
                    calculate_numerical_gradient(
                        model=model,
                        parameter=parameter,
                        parameter_index=parameter_index,
                        features=features,
                        targets=targets,
                        loss_function=loss_function,
                    )
                )
            absolute_errors = np.abs(
                analytical_gradient - numerical_gradient
            )

            denominators = np.maximum(
                np.abs(analytical_gradient)
                + np.abs(numerical_gradient),
                1e-12,
            )
            relative_errors = (
                absolute_errors / denominators
            )
            parameter_max_absolute_error = float(
                np.max(absolute_errors)
            )
            parameter_max_relative_error = float(
                np.max(relative_errors)
            )
            maximum_absolute_error = max(
                maximum_absolute_error,
                parameter_max_absolute_error,
            )
            maximum_relative_error = max(
                maximum_relative_error,
                parameter_max_relative_error,
            )

            parameter_passed = (
                parameter_max_relative_error
                < RELATIVE_ERROR_TOLERANCE
            )

            if not parameter_passed:
                all_gradients_passed = False

            total_parameters_checked += parameter.size

            status = "PASS" if parameter_passed else "FAIL"

            print(
                f"Layer {layer_number} | {parameter_name:<7} "
                f"| Shape: {str(parameter.shape):<8} "
                f"| Max relative error: "
                f"{parameter_max_relative_error:.3e} "
                f"| {status}"
            )
            # Print detailed comparisons for individual values.
            for parameter_index in np.ndindex(parameter.shape):
                analytical_value = float(
                    analytical_gradient[parameter_index]
                )
                numerical_value = float(
                    numerical_gradient[parameter_index]
                )
                error = float(
                    relative_errors[parameter_index]
                )
                print(
                    f"    Index {str(parameter_index):<8}"
                    f" Analytical: {analytical_value: .8e}"
                    f" | Numerical: {numerical_value: .8e}"
                    f" | Relative error: {error:.3e}"
                )
    # FINAL SUMMARY
    print("\nFINAL SUMMARY")
    print("-" * 72)
    print(
        f"Parameters checked      : {total_parameters_checked}"
    )
    print(
        f"Maximum absolute error  : "
        f"{maximum_absolute_error:.3e}"
    )
    print(
        f"Maximum relative error  : "
        f"{maximum_relative_error:.3e}"
    )
    print(
        f"Relative error tolerance: "
        f"{RELATIVE_ERROR_TOLERANCE:.3e}"
    )
    if all_gradients_passed:
        print("\nRESULT: PASS")
        print(
            "All checked parameter gradients agree within "
            "the specified tolerance."
        )
    else:
        print("\nRESULT: FAIL")
        print(
            "Some gradients differ beyond the specified "
            "tolerance. Investigate the affected parameters."
        )

    return all_gradients_passed
#execution
def main():
    np.random.seed(RANDOM_SEED)
    features = np.array(
        [0.37],
        dtype=np.float64,
    )
    targets = np.array(
        [1.0],
        dtype=np.float64,
    )
    model = build_model()
    loss_function = BinaryCrossEntropy()
    initial_loss = calculate_loss(
        model,
        features,
        targets,
        loss_function,
    )
    print("=" * 72)
    print("RAWGRAD: NUMERICAL GRADIENT CHECKING")
    print("=" * 72)
    print(f"Input features : {features}")
    print(f"Target         : {targets}")
    print(f"Initial loss   : {initial_loss:.8f}")
    print(f"Epsilon        : {EPSILON:.1e}")
    print(
        f"Tolerance      : "
        f"{RELATIVE_ERROR_TOLERANCE:.1e}"
    )
    check_gradients(
        model=model,
        features=features,
        targets=targets,
        loss_function=loss_function,
    )
if __name__ == "__main__":
    main()