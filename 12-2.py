import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

'''
실행 할 때 마다 결과가 달라지는 것을 방지
random 컴퓨터의 현재 시간의 영향을 받음 => seed
seed가  컴퓨터의 현재 시간의 영향을 받아서 변경되니까, seed 값을 고정하면
랜덤값이 안 바뀌겠죠?
reproducability(재현성)
'''
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

# 데이터 불러오기
dataraw = pd.read_csv('data/BTC-USD.csv'
                      , index_col='Date'
                      , parse_dates=['Date'])
# 정렬된 날짜 기준으로  data 불러오기
dataset = pd.DataFrame(dataraw['Close'])
# 불러온 데이터(정렬된 날짜 기준으로) 종가만 가져옴

# 데이터 정규화
scaler = MinMaxScaler()
# Zero-mean Unit Variance 라는 정규화를 사용
# MinMax 정규화를 이용 (t-min)/(max-min)
dataset_norm = dataset.copy()
# 나중에 기존 데이터를 활용할텐데, 정규화를 통해 값이 변한다면
# 확인이 불가능하겠죠?
dataset_norm['Close'] = scaler.fit_transform(dataset[['Close']])

totaldata = dataset.values
totaldatatrain = int(len(totaldata) * 0.7)
totaldataval = int(len(totaldata) * 0.1)
training_set = dataset_norm[0:totaldatatrain]
val_set = dataset_norm[totaldatatrain:totaldatatrain + totaldataval]
test_set = dataset_norm[totaldatatrain + totaldataval:]


def create_sliding_windows(data, len_data, lag):
    x, y = [], []
    for i in range(lag, len_data):
        x.append(data[i - lag:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


lag = 2

array_training_set = np.array(training_set)
array_val_set = np.array(val_set)
array_test_set = np.array(test_set)

x_train, y_train = create_sliding_windows(array_training_set, len(array_training_set), lag)
x_val, y_val = create_sliding_windows(array_val_set, len(array_val_set), lag)
x_test, y_test = create_sliding_windows(array_test_set, len(array_test_set), lag)

x_train, y_train = (torch.tensor(x_train, dtype=torch.float32),
                    torch.tensor(y_train, dtype=torch.float32))
x_val, y_val = (torch.tensor(x_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32))
x_test, y_test = (torch.tensor(x_test, dtype=torch.float32),
                  torch.tensor(y_test, dtype=torch.float32))


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1,
                          batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.lstm(x)
        out = self.fc(h[-1])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=3,
                          batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.gru(x)
        out = self.fc(h[-1])
        return out


input_size = 1
hidden_size = 64
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 1
batch_size = 256

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train.unsqueeze(-1))
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        val_outputs = model(x_val.unsqueeze(-1))
        val_loss = criterion(val_outputs.squeeze(), y_val)
        print(f"Epoch [{epoch + 1}/{epochs},"
              f"loss: {loss.item():.4f},"
              f"ValLoss : {val_loss.item():.4f}")

model.eval()
y_pred = model(x_test.unsqueeze(-1)).detach().numpy()
y_pred_inver_norm = scaler.inverse_transform(y_pred.squeeze(0))


def rmse(dataset, datapred):
    return np.sqrt(np.mean((datapred - dataset) ** 2))


def mape(dataset, datapred):
    return np.mean(np.abs((dataset - datapred) / dataset)) * 100


dataset = dataset['Close'][totaldatatrain + totaldataval + lag:].values
print('RMSE:', rmse(dataset, y_pred_inver_norm))
print('MAPE:', mape(dataset, y_pred_inver_norm))

plt.figure(figsize=(10, 4))
plt.plot(dataset, label="Data Test", color='red')
plt.plot(y_pred_inver_norm, label="predictions", color='blue')
plt.title('bt')
plt.xlabel('Day')
plt.ylabel('price')
plt.legend()
plt.show()