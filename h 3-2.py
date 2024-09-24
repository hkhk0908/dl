import numpy as np

class Layer_Dense:
    def __init__(self):
        self.output = None

    def softmax(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return probabilities

    @staticmethod
    def categorical_cross_entropy(predictions, targets):
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)

        if targets.ndim == 1:
            correct_confidences = predictions[np.arange(len(predictions)), targets]
        else:
            correct_confidences = np.sum(predictions * targets, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

logits = np.array([
    [2.0, 1.0, 0.1],
    [1.0, 3.0, 0.2],
    [0.2, 0.9, 4.0]
])

targets = np.array([0, 1, 2])

layer = Layer_Dense()
softmax_outputs = layer.softmax(logits)

loss = Layer_Dense.categorical_cross_entropy(softmax_outputs, targets)
print("Categorical Cross-Entropy Loss:", loss)