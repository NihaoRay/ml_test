import math
import torch
from torch import nn
from d2l import torch as d2l


def masked_softmax(X, valid_lens):
    """通过在最后⼀个轴上掩蔽元素来执⾏softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape

        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[0])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))


# 加性注意⼒
class AdditiveAttention(nn.Module):
    """加性注意⼒"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有⼀个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


class AddAttention(nn.Module):
    """加性注意⼒"""
    def __init__(self, dropout, **kwargs):
        super(AddAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        # queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries + keys
        features = torch.tanh(features)
        # self.w_v仅有⼀个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        # scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(features, torch.tensor([queries.shape[1]]))
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.matmul(self.dropout(self.attention_weights), values.permute(1, 0))


# queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
queries = torch.ones((10, 8))
print(queries)


attent = AddAttention(0.3)


# values的⼩批量，两个值矩阵是相同的
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)

valid_lens = torch.tensor([2, 6])

# attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)

attent.eval()

print(attent(queries, queries, queries))

# k = torch.tensor([[ [1, 2],
#                     [3, 4],
#                     [5, 6],
#                     [7, 8] ],
#
#                   [ [9, 10],
#                     [11, 12],
#                     [13, 14],
#                     [15, 16] ],
#
#                   [ [17, 18],
#                     [19, 20],
#                     [21, 22],
#                     [22, 23] ]])
#
#
# v = torch.tensor([[1, 2],
#                   [4, 5],
#                   [7, 8]])
#
# res = torch.bmm(v, k)
#
# print(res)


