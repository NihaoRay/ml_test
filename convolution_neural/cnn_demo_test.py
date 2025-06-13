import torch
from torch import nn


# 为了⽅便起⻅，我们定义了⼀个计算卷积层的函数
# 此函数初始化卷积层权重，并对输⼊和输出提⾼和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这⾥的（1，1）表⽰批量⼤⼩和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量⼤⼩和通道
    return Y.reshape(Y.shape[2:])


conv2d = nn.Conv2d(1, 1, kernel_size=5, padding=2)
X = torch.rand(size=(28, 28))
print(comp_conv2d(conv2d, X).shape)


