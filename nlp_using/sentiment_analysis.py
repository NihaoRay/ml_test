import os
import torch
from torch import nn
from d2l import torch as d2l
import BiRNN as birnn

# d2l.DATA_HUB['aclImdb'] = ( 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
#                             '01ada507287d82875905620988597833ad4e0903')
# data_dir = d2l.download_extract('aclImdb', 'aclImdb')

data_dir = '../data/aclImdb'

def read_imdb(data_dir, is_train):
    """读取IMDb评论数据集文本序列和标签"""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)

        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)

    return data, labels


# train_data = read_imdb(data_dir, is_train=True)
# print('训练集数⽬：', len(train_data[0]))
# for x, y in zip(train_data[0][:3], train_data[1][:3]):
#     print('标签：', y, 'review:', x[0:60])



# train_tokens = d2l.tokenize(train_data[0], token='word')
# vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])

# d2l.set_figsize()
# d2l.plt.xlabel('# tokens per review')
# d2l.plt.ylabel('count')
# d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50))
# # d2l.plt.show()

# num_steps = 500 # 序列长度
# train_features = torch.tensor([d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])

# print(train_features.shape)

# 创建数据迭代器
# train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)

# for X, y in train_iter:
#     print('X:', X.shape, ', y:', y.shape)
#     break
#
# print('小批量数目：', len(train_iter))



def load_data_imdb(data_dir, batch_size, num_steps = 500):
    """返回数据迭代器和IMDb评论数据集的词表"""
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)

    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')

    vocab = d2l.Vocab(train_tokens, min_freq=5)

    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])

    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])

    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)

    return train_iter, test_iter, vocab



batch_size = 64
train_iter, test_iter, vocab = load_data_imdb(data_dir, batch_size)

torch.save(vocab, 'sentiment_analysis.vocab')

embed_size, num_hiddens, num_layers = 100, 100, 2
devices = d2l.try_all_gpus()
net = birnn.BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
net.apply(init_weights)

glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]

# 我们使用这些预训练的词向量来表示评论中的词元，并且在训练期间不要更新这些向量
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False

lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")

d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

torch.save(net.state_dict(), 'sentiment_analysis.pt')










