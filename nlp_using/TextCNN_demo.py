import torch
from torch import nn

def corr1d(X, K):
    w = K.shape[0]
    Y = torch.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y


X, K = torch.tensor([0, 1, 2, 3, 4, 5, 6]), torch.tensor([1, 2])

# print(corr1d(X, K))

def corr1d_multi_in(X, K):
    # ⾸先，遍历'X'和'K'的第0维（通道维）。然后，把它们加在⼀起
    for x, k in zip(X, K):
        print(corr1d(x, k))

    return sum(corr1d(x, k) for x, k in zip(X, K))

X = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
                    [1, 2, 3, 4, 5, 6, 7],
                    [2, 3, 4, 5, 6, 7, 8]])

K = torch.tensor([[1, 2], [3, 4], [-1, -3]])

print(corr1d_multi_in(X, K))


