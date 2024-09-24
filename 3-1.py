import numpy as np
import matplotlib.pyplot as plt
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        '''
        :param n_inputs: 입력의 개수
        :param n_neurons: 뉴런의 개수
        '''
        self.weights = np.zeros((n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Linear:
    def forward(self, inputs):
        self.output = inputs

x = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y = np.sin(x)

dense1 = Layer_Dense(1, 8)
dense2 = Layer_Dense(8, 8)
dense3 = Layer_Dense(8, 1)

dense1.weights = np.array([[3.0], [2.0], [3.0], [2.0], [3.0], [2.0], [3.0], [2.0]]).T
dense1.biases = np.array([[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]])

dense2.weights = np.random.randn(8, 8) * 0.1
dense2.biases = np.ones((1, 8)) * 0.2

dense3.weights = np.random.randn(8, 1) * 0.2
dense3.biases = np.ones((1, 1))

activation1 = Activation_ReLU()
activation2 = Activation_ReLU()
activation3 = Activation_Linear()

dense1.forward(x)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

dense3.forward(activation2.output)
activation3.forward(dense3.output)

plt.plot(x, y, color='blue', label='sin(x)')
plt.plot(x, activation3.output, color='red', label='Neural Network Output')
plt.legend()
plt.show()


