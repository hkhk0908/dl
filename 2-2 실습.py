import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = (np.random.uniform
                        (0, 1, (n_inputs, n_neurons)))
        self.biases = np.random.uniform(0, 1, (1, n_neurons))

    def forward(self, inputs):
        return np.dot(inputs,self.weights) + self.biases

X, y = spiral_data(200, 3)

layer1 = Layer_Dense(2, 2)
output1 = layer1.forward(X)
print("첫 번째 레이어 출력:")
print(output1)

layer2 = Layer_Dense(2, 5)
output2 = layer2.forward(output1)
print("두 번째 레이어 출력:")
print(output2)