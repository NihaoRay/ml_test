import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size = 1
seq_len = 3
input_size = 4 # 也就是one_hot的长度
hidden_size = 2

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)

for idx, input in enumerate(dataset):
    print('=' * 20, idx, '='*20)
    print('Input size:', input.shape)

    hidden = cell(input, hidden)

    print('output size:', hidden.shape)
    print(hidden)














