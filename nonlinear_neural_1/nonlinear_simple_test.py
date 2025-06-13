import torch
import torchvision
from IPython import display
from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

# 定义网络模型
net = nn.Sequential(nn.Flatten(),
              nn.Linear(784, 256),
              nn.ReLU(),
              nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

# 损失函数
loss = nn.CrossEntropyLoss(reduction='none')
batch_size, num_epochs, lr = 256, 10, 0.1

# 加载数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 梯度优化函数
updater = torch.optim.SGD(net.parameters(), lr=lr)


d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

d2l.predict_ch3(net, test_iter)
d2l.plt.show()













