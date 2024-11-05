import torch
import torch.nn as nn
import numpy as np
import torchvision

#x = torch.tensor(1.,requires_grad=True)
#w = torch.tensor(2.,requires_grad=True)
#b = torch.tensor(3.,requires_grad=True)

#y = x*w + b

#y.backward()

#print(x.grad,w.grad,b.grad)


#x = torch.randn(10,3)
#y = torch.randn(10,2)

#linear = nn.Linear(3,2)
#print('w: ',linear weight)
#print('b: ',linear.bias)

#criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(linear.parameters(),lr = 0.01)

#for i in range(100):
    #pred = linear(x)
    #loss = optimizer(pred, y)

#pred = linear(x)

#loss = criterion(pred, y)
#print('loss :', loss.item())

#loss.backward()

#print('dL/dw:', linear.weight.grad)
#print('dL/db', linear.bias.grad)

#optimizer.step()

#pred = linear(x)
#loss = criterion(pred, y)
#print(loss.item())

#x = np.array([[1,2],[3,4]])
#y = torch.from_numpy(x)
#z = y.numpy()
#print("A")

#train_dataset = torchvision.datasets.CIFAR10(
#    root='./data',
#    train=True,
#    transform=torchvision.transforms.ToTensor(),
#    download=True
#)

#image, label = train_dataset[0]
#print(image.size)
#print(label)

#train_loader = torch.utils.data.DataLoader(
#    dataset= train_dataset,
#    batch_size=64,
#    shuffle=True
#)

#data_iter = iter(train_loader)

#images, labels = data_iter.next()

#for images, labels in train_loader:
#    pass

#class CustomDataset(torch.utils.data.Dataset):
#    def __init__(self):
#        x = 입력데이터 뭉치
#        y = 정답 데이터 뭉치

#    def __getitem__(self,index):
#        return x[index], y[index]
#
#    def __len__(self):
#        return len(self.x)

resnet = torchvision.models.resnet18(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features, 100)

torch.save(resnet,'model.ckpt')
model = torch.load('model.ckpt')
