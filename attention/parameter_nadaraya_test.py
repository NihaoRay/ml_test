import torch
from torch import nn
from d2l import torch as d2l


# 训练样本数
n_train = 50
# n_train = 3
# 排序后的训练样本
x_train, _ = torch.sort(torch.rand(n_train) * 5)
# x_train = torch.tensor([1, 2, 3])

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
# y_train = torch.tensor([4, 5, 6])

class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(-((queries - keys) * self.w) ** 2 / 2, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)


# X_tile的形状:(n_train，n_train)，每⼀⾏都包含着相同的训练输⼊
X_tile = x_train.repeat((n_train, 1))

# Y_tile的形状:(n_train，n_train)，每⼀⾏都包含着相同的训练输出
Y_tile = y_train.repeat((n_train, 1))

# keys的形状:('n_train'，'n_train'-1)
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

# values的形状:('n_train'，'n_train'-1)
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))


net = NWKernelRegression()
# 平⽅损失函数
loss = nn.MSELoss(reduction='none')
# 随机梯度下降
trainer = torch.optim.SGD(net.parameters(), lr=0.5)

# animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    y_hat = net(x_train, keys, values)
    l = loss(y_hat, y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    # animator.add(epoch + 1, float(l.sum()))


x_test = torch.arange(0, 5, 0.1)
y_truth = f(x_test)
n_test = len(x_test)


def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)
    d2l.plt.show()

keys = x_train.repeat((n_test, 1))
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)

# d2l.plt.show()
