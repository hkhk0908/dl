import numpy as np

# 입력의 개수를 기반으로 가중치를 랜덤하게 초기화하는 함수
def init_weight(inputs):
    # -1과 1 사이에서 랜덤하게 가중치 초기화
    weights = np.random.uniform(-1, 1, size=len(inputs))
    return weights

# 입력값과 가중치, 바이어스를 받아 결과를 계산하는 함수
def cal(inputs, weights, bias):
    # 입력과 가중치의 내적(dot product)을 계산한 후 바이어스를 더함
    output = np.dot(inputs, weights) + bias
    return output

# 뉴런의 개수를 입력받고 계산하는 함수
def cal_neuron(num_neuron, inputs):
    outputs = []
    for _ in range(num_neuron):
        # 각 뉴런에 대해 가중치 초기화
        weights = init_weight(inputs)
        bias = np.random.uniform(-1, 1)  # 바이어스도 랜덤하게 초기화
        # 뉴런의 출력 계산
        output = cal(inputs, weights, bias)
        outputs.append(output)
    return outputs

# 사용 예시
inputs = [0.5, -0.2, 0.1]  # 입력 데이터 예시
num_neuron = 3  # 뉴런의 개수

# 뉴런 계산
outputs = cal_neuron(num_neuron, inputs)
print(outputs)