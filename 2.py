import random

# 입력의 개수에 따라 랜덤한 가중치를 생성하는 함수
def init_weights(inputs):
    weights = []
    for i in range(len(inputs)):
        weights.append(random.uniform(-1, 1))
    return weights

# 입력값, 가중치, 바이어스를 받아 결과를 계산하는 함수
def cal(inputs, weights, bias):
    output = 0
    for i in range(len(inputs)):
        output += inputs[i] * weights[i]
    output += bias
    return output

# 뉴런의 개수를 입력받고 계산하는 함수
def cal_neuron(num_neuron, inputs):
    outputs = []
    for _ in range(num_neuron):
        weights = init_weights(inputs)
        bias = random.uniform(-1, 1)
        output = cal(inputs, weights, bias)
        outputs.append(output)
    return outputs

# 사용 예시
inputs = [1.0, 2.0, 3.0]
num_neuron = 3

# 뉴런의 출력 계산
outputs = cal_neuron(num_neuron, inputs)

# 결과 출력
print("Outputs:", outputs)