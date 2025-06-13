import collections
import math
import torch
from torch import nn
from d2l import torch as d2l

# ⽤于序列到序列学习的循环神经⽹络编码器
class Seq2SeqEncoder(d2l.Encoder):
    """⽤于序列到序列学习的循环神经⽹络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌⼊层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第⼀个轴对应于时间步
        x_steps = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(x_steps)
        # output的形状:(num_steps, batch_size, num_hiddens)
        # state[0]的形状:(num_layers, batch_size, num_hiddens)
        return output, state


# encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# X = torch.zeros((4, 7), dtype=torch.long)
# output, state = encoder(X)
# print(output.shape, state.shape)

class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        # 这个其实RNN将隐藏层的数据输出out，外界了一个MLP将隐藏层H转为out
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # ⼴播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state[0]的形状:(num_layers,batch_size,num_hiddens)
        return output, state

# decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# decoder.eval()
# state = decoder.init_state(encoder(X))
# output, state = decoder(X, state)
# print(output.shape, state.shape)

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,  device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# sequence_mask(X, torch.tensor([1, 2]))

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        # unweighted_loss = cross_entropy(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

# loss = MaskedSoftmaxCELoss()
# ret = loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0]))
# print(ret)



