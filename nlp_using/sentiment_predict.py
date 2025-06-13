import os
import torch
from torch import nn
from d2l import torch as d2l
import BiRNN as birnn


vocab = torch.load('sentiment_analysis.vocab')
embed_size, num_hiddens, num_layers = 100, 100, 2
devices = d2l.try_all_gpus()

net = birnn.BiRNN(len(vocab), embed_size, num_hiddens, num_layers).to(devices[0])

net.load_state_dict(torch.load('sentiment_analysis.pt'))

net.eval()

glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]

# 我们使用这些预训练的词向量来表示评论中的词元，并且在训练期间不要更新这些向量
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False

def predict_sentiment(net, vocab, sequence):
    """预测⽂本序列的情感"""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'


print(predict_sentiment(net, vocab, 'every one doese not like'))
print(predict_sentiment(net, vocab, 'this movie is so bad'))
print(predict_sentiment(net, vocab, 'oh, Its not good, the plot does not work'))