import torch
import torchvision
from IPython import display
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from d2l import torch as d2l

# 对每个通道执⾏互相关操作
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在⼀起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

# X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
# [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
# K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

# print(corr2d_multi_in(X, K))

# print(K)


def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输⼊“X”执⾏互相关运算。
    # 最后将所有结果都叠加在⼀起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

# K = torch.stack((K, K + 1, K + 2), 0)
# print(K.shape)
# print(K)

# print(corr2d_multi_in_out(X, K))

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size = 5, padding = 3), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

# batch_size, channels, height, width
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        # 一次获取batch_size数量的数据集合
        for X, y in data_iter:
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    # 初始化模型中的权重
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    # 初始化梯度下降算法
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 交叉损失
    loss = nn.CrossEntropyLoss()

    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])

    timer, num_batches = d2l.Timer(), len(train_iter)
    # 执行训练
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            # 梯度首先清零，先清和后清是一样的，在首次执行的时候梯度时为零的
            optimizer.zero_grad()

            y_hat = net(X)
            l = loss(y_hat, y)

            l.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))


        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


lr, num_epochs = 0.9, 8
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# 根据训练完成优化后的w1, w2, b1, b2的参数，然后调用网络进行计算
d2l.predict_ch3(net, test_iter)
d2l.plt.show()











