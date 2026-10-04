
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    def step(self, model):
        for layer in model.layers:
            if not hasattr(layer, "parameters"):
                continue
            if not hasattr(layer, "gradient"):
                continue
            parameters = layer.parameters()
            gradients = layer.gradient()
            if len(parameters) != len(gradients):
                raise ValueError(
                    "Each parameter must have a matching gradient."
                )
            for parameter, gradient in zip(parameters, gradients):
                parameter -= self.learning_rate * gradient
class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
    def step(self, model):
        for layer_index, layer in enumerate(model.layers):
            if not hasattr(layer, "parameters"):
                continue
            if not hasattr(layer, "gradient"):
                continue
            parameters = layer.parameters()
            gradients = layer.gradient()
            if len(parameters) != len(gradients):
                raise ValueError(
                    "Each parameter must have a matching gradient."
                )
            if layer_index not in self.velocities:
                self.velocities[layer_index] = [
                    parameter * 0.0 for parameter in parameters
                ]
            for index, (parameter, gradient) in enumerate(
                zip(parameters, gradients)
            ):
                velocity = self.velocities[layer_index][index]
                velocity = (
                    self.momentum * velocity
                    - self.learning_rate * gradient
                )
                self.velocities[layer_index][index] = velocity
                parameter += velocity