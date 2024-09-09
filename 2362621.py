import numpy as np

inputs = [[2.0, 3.0, 4.0, 3.5],
          [3.0, 6.0,-2.0, 3.0],
          [-2.5,3.7,4.3,-1.8]]
weights = [[1.2, 1.8, -1.5, 2.0],
           [1.5,-1.91,1.26,-1.5],
           [1.26,-1.27,1.17,1.87]]

biases = [2.0,3.0,0.5]

layers_outputs = np.dot(inputs, np.array(weights).T)+biases
print(layers_outputs)