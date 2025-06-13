import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l


# 打印全部长度
# torch.set_printoptions(threshold=torch.inf)


num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 这⾥应⽤了⼴播机制

X = torch.normal(0, 1, (2, 5))
# print(X)
X_prob = softmax(X)
# print(X_prob)
# X_prob.sum(1)

def net(X):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)

y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# 就是表达第"0"行我挑选第"0"个元素，第"1"行我挑选第"2"个元素
y_hat[[0, 1], y]

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
# cross_entropy(y_hat, y)

print(y_hat[range(len(y_hat)), y])



