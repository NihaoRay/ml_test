import torch
import torchvision
from IPython import display
from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

# 加载数据集合
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 定义网络模型
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]

# relu激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 定义网络模型
def net(X):
    # reshape将每个⼆维图像转换为⼀个⻓度为num_inputs的向量
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1) # 这⾥“@”代表矩阵乘法
    return (H @ W2 + b2)

# 交叉损失
loss = nn.CrossEntropyLoss(reduction='none')

# 梯度下降训练
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)

# 执行训练
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

# 根据训练完成优化后的w1, w2, b1, b2的参数，然后调用网络进行计算
d2l.predict_ch3(net, test_iter)
d2l.plt.show()













