import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, initialize_method='random'):
        if initialize_method == 'random':
            self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))
        elif initialize_method == 'xavier':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
        elif initialize_method == 'he':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        else:
            raise ValueError("지원하지 않는 초기화 방법입니다.")

        self.biases = np.random.uniform(0, 1, (1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        self.output = np.maximum(0, self.output)

X, y = spiral_data(200, 3)

layer1 = Layer_Dense(2, 5, initialize_method='random')
layer1.forward(X)
print("첫 번째 레이어 출력:", layer1.output)

layer2 = Layer_Dense(5, 3, initialize_method='xavier')
layer2.forward(layer1.output)
print("두 번째 레이어 출력:", layer2.output)


