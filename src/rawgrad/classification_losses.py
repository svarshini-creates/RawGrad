import numpy as np
class BinaryCrossEntropy:
    def forward(self, predictions, targets):
        self.predictions = np.clip(
            predictions,
            1e-7,
            1.0 - 1e-7,
        )
        self.targets = targets
        loss = -(
            targets * np.log(self.predictions)
            + (1.0 - targets)
            * np.log(1.0 - self.predictions)
        )
        return np.mean(loss)
    def backward(self):
        gradients = -(
            self.targets / self.predictions
            - (1.0 - self.targets)
            / (1.0 - self.predictions)
        )
        return gradients / self.predictions.size
class CategoricalCrossEntropy:
    def forward(self, predictions, targets):
        self.predictions = np.clip(
            predictions,
            1e-7,
            1.0 - 1e-7,
        )
        self.targets = targets
        sample_losses = -np.sum(
            targets * np.log(self.predictions),
            axis=1,
        )
        return np.mean(sample_losses)
    def backward(self):
        return -(
            self.targets / self.predictions
        ) / self.targets.shape[0]