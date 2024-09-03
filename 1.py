import random

def init_weights(inputs):

    for i in range(len(inputs)):
        random.append(random.uniform(-1, 1))
    return weights

inputs = [1.0, 2.0, 3.0]
weights = [0.2,0.8,-0.5]
bias = 2.0

output = \
    inputs[0]*weights[0] +\
    inputs[1]*weights[1] +\
    inputs[2]*weights[2] +\
    bias

print(output)