import torch
import torch.nn.functional as f
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):  # 初始化信息
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 一层隐藏层神经网络，输入信息，输出信息（神经元个数）
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 一层预测层神经网络，输入信息，输出信息（神经元个数）

    def forward(self, x):  # 前向传递，x为输入信息
        x = f.relu(self.hidden(x))  # 出了隐藏层要激活
        x = self.predict(x)
        return x


model = Net(1, 10, 1)

model.load_state_dict(torch.load('linara-test.pt'))

x = torch.tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.7,0.8,0.9,0.7,0.8,0.9,0.7,0.8,0.9,0.7,0.8,0.9,0.7,0.8,0.9])
x = torch.reshape(x, (x.shape[0], 1))
print(model(x))

print('================')
print('y')
y = x.pow(2) + 0.2 * torch.rand(x.size())  # y=x*2，但是还要加上波动

print(y)

