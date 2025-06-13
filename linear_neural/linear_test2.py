import random
import torch
from d2l import torch as d2l

y_data = torch.tensor([[ 0.9024],
        [ 4.9539],
        [ 1.2366],
        [13.4016],
        [ 9.7447],
        [ 2.5708],
        [ 9.1989],
        [ 4.5129],
        [ 5.9701],
        [ 0.6199]])
X_data = torch.tensor([[-2.5401, -0.5274],
        [-0.6940, -0.6292],
        [ 0.5387,  1.1866],
        [ 1.6646, -1.7324],
        [ 1.5421, -0.7289],
        [-0.6030,  0.1265],
        [-0.8599, -1.9786],
        [-0.3054, -0.2693],
        [-0.4930, -0.8057],
        [ 0.3335,  1.2483]])


batch_size = 10

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
num_epochs = 1 #迭代周期个数

# 初始化模型参数
# w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
w = torch.tensor([[ 0.0091],  [-0.0009]], requires_grad=True)


for epoch in range(num_epochs):
    y_hat = linreg(X_data, w, b) # forward 前馈计算
    l = squared_loss(y_hat, y_data) # X和y的⼩批量损失
    # 因为l形状是(batch_size,1)，⽽不是⼀个标量。l中的所有元素被加到⼀起，
    # 并以此计算关于[w,b]的梯度
    l.sum().backward()

    print('y:', y_data)
    print('x:', X_data)
    print('y_hat:', y_hat)
    print('loss:', l)
    print('#########################')
    print('w:', w, '\n b:', b)
    print('w grad:', w.grad, '\n', 'b grad:', b.grad)

    sgd([w, b], lr, batch_size) # 使用参数的梯度更新参数

    with torch.no_grad():
        train_l = squared_loss(linreg(X_data, w, b), y_data)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        print(w, '\n', b)



