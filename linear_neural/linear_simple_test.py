import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

#----------------------------------------------------------------------------#
# 单层的线性回归实现
# 模型：f(x) = [x1, x2][w1, w2]' + b
# 损失函数：MSELoss: sum((y_hat[i] - y[i])^2) / batch_size，这里i长度是batch_size
# 梯度更新方式：小批量每次更新迭代的损失公式计算
#----------------------------------------------------------------------------#

# 生成模拟数据
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 加载数据
def load_array(data_arrays, batch_size, is_train=True): #@save
    """构造⼀个PyTorch数据迭代器"""
    dataset = data.TensorDataset( * data_arrays)
    # 相当于zip函数
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))

# 定义模型
# 第⼀个指定输⼊特征形状，即2，第⼆个指定输出特征形状，输出特征形状为单个标量，因此为1
net = nn.Sequential(nn.Linear(2, 1))
# 重写带训练更新的参数值
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 迭代生命周期⾥
num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        print('epoch, loss, w, b:', epoch ,  l.item() , net[0].weight.data , net[0].bias.data)
        # 在反向自动微分计算梯度之前，需要进行梯度清零
        trainer.zero_grad()
        l.backward()
        # 调用优化算法，进行参数更新
        trainer.step()

    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')


w = net[0].weight.data
print('w:', w, ', w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b:', b, ', b的估计误差：', true_b - b)












