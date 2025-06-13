import math
import torch
from torch import nn
from d2l import torch as d2l

num_hiddens, num_heads = 6, 1
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)

# print(attention.eval())

batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
x = torch.randn((batch_size, num_queries, num_hiddens))

print(f'x.shape:{x.shape}')
print('--------')
print(x)

# print(attention(X, X, X, valid_lens).shape)
attention_result = attention(x, x, x, valid_lens=None)
print(f'attention_result:{attention_result.shape} \n {attention_result}')

dot_attention = d2l.DotProductAttention(0.3)

y = torch.randn(3, 2).unsqueeze(2)
print(y)


print(dot_attention(y, y, y))




class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 相同形状的位置嵌入矩阵P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)



encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()

X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]

d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)', figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
d2l.plt.show()

