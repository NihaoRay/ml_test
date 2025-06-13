import math
import torch
from torch import nn
from d2l import torch as d2l

batch_size, max_window_size, num_noise_words = 512, 5, 5

data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size, num_noise_words)
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)

# print(f'Parameter embedding_weight ({embed.weight.shape}, 'f'dtype={embed.weight.dtype})')
# x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(embed(x))

def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

# print(skip_gram(torch.ones((2, 1), dtype=torch.long), torch.ones((2, 4), dtype=torch.long), embed, embed).shape)

class SigmoidBCELoss(nn.Module):
    # 带掩码的⼆元交叉熵损失
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()

pred = torch.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))


embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))


def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])

    # 规范化的损失之和，规范化的损失数
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            # 这里的label是context的词数向量，一个labels变量将上下⽂词与噪声词分开
            l = ( loss(pred.reshape(label.shape).float(), label.float(), mask) / mask.sum(axis=1) * mask.shape[1])

            l.sum().backward()

            optimizer.step()
            metric.add(l.sum(), l.numel())

            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,(metric[0] / metric[1],))

        print(f'loss {metric[0] / metric[1]:.3f}, 'f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')


lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)

def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9)

    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')

    sim_token = []
    for i in topk[1:]: # 删除输⼊词
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')
        sim_token.append(vocab.to_tokens(i))
    return sim_token

get_similar_tokens('chip', 3, net[0])
