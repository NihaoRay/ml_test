import random
import torch
from d2l import torch as d2l

# 打印全部长度
# torch.set_printoptions(threshold=torch.inf)

# 生成数据集
def synthetic_data(w, b, num_examples):
    #@save
    """⽣成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    # y.reshape((-1, 1))的含义：若y中含有n个元素，则把y reshape成shape为n*1的列向量
    return X, y.reshape((-1, 1))


# 获得数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0],'\nlabel:', labels[0])


batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break



# 定义模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y): #@save
    """均⽅损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法 学习速率lr
def sgd(params, lr, batch_size): #@save
    """⼩批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.03 # 学习率
num_epochs = 3 #迭代周期个数

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        y_hat = linreg(X, w, b) # forward 前馈计算
        l = squared_loss(y_hat, y) # X和y的⼩批量损失
        # 因为l形状是(batch_size,1)，⽽不是⼀个标量。l中的所有元素被加到⼀起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()

        print('y_hat:', y_hat)
        print('y:', y)
        print('x:', X)
        print('#########################')
        print('w:', w, '\n b:', b)
        print('w grad:', w.grad, '\n', 'b grad:', b.grad)

        print('check b grad:', (y_hat - y))
        print('check w grad:', (y_hat - y)*b)

        sgd([w, b], lr, batch_size) # 使用参数的梯度更新参数


    with torch.no_grad():
        train_l = squared_loss(linreg(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        print(w, '\n', b)


# print(features[:, (1)].detach().numpy())


# d2l.set_figsize()
# d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.show()

x = torch.tensor([2, 1, 3, 5, 4])
y = torch.tensor([4, 2, 6, 10, 8])

a = x + y

arr = []
for x_i, y_i in x, y:
    b = x_i + y_i
    arr.append(b)

arr = x + y

# mn  x nN = mN

print(arr)

# x = torch.tensor([[2, 4], [1, 2], [3, 6], [5,10], [4, 8]])
#
# print(x[:, (2)])
#
# d2l.set_figsize()
# d2l.plt.scatter(x.detach().numpy(), y.detach().numpy(), 1)
# d2l.plt.show()


