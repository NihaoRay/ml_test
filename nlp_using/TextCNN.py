import torch
from torch import nn
from d2l import torch as d2l


batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 这个嵌入层不需要训练
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 最大时间汇聚层没有参数，因此可以共享此实例
        # 对每个通道进行池化，比如本例子中(64, 100, 498) 表示 batch_size, 通道数, 宽度 最后会变成 (64, 100, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pool2 = nn.AdaptiveAvgPool1d(2) # 仅仅作为测试用的，是对每个通道进行池化
        # 创建多个⼀维卷积层
        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # 沿着向量维度将两个嵌入层连结起来，
        # 每个嵌入层的输出形状都是（批量大小，词元数量，词元向量维度）连结起来
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # 根据⼀维卷积层的输入格式，重新排列张量，以便通道作为第2维
        embeddings = embeddings.permute(0, 2, 1)
        # 每个⼀维卷积层在最大时间汇聚层合并后，获得的张量形状是（批量大小，通道数，1）
        # 删除最后⼀个维度并沿通道维度连结
        conv_cnn = []
        for conv in self.convs:
            embed_conv = conv(embeddings)
            conv_cnn.append(torch.squeeze(self.relu(self.pool(embed_conv)), dim=-1))

        encoding = torch.cat(conv_cnn, dim=1)
        # encoding = torch.cat([torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1) for conv in self.convs],
        #                      dim=1)
        outputs = self.decoder(self.dropout(encoding))

        return outputs


embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)


def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)


glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]

net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False


lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)




print(d2l.predict_sentiment(net, vocab, 'this movie is so great'))
print(d2l.predict_sentiment(net, vocab, 'this movie is so bad'))
print(d2l.predict_sentiment(net, vocab, 'oh, Its not good, the plot does not work'))
print(d2l.predict_sentiment(net, vocab, 'every one doese not like'))

d2l.plt.show()