import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, initialize_method='random'):
        if initialize_method == 'random':
            self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))
        elif initialize_method == 'xavier':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1. / n_inputs)
        elif initialize_method == 'he':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2. / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        output = np.dot(inputs, self.weights) + self.biases
        return np.maximum(0, output)  # ReLU 함수 적용

X, y = spiral_data(200, 3)

layer1 = Layer_Dense(2, 2, initialize_method='he')
output1 = layer1.forward(X)
print("첫 번째 레이어 출력:")
print(output1)

layer2 = Layer_Dense(2, 5, initialize_method='xavier')
output2 = layer2.forward(output1)
print("두 번째 레이어 출력:")
print(output2)

