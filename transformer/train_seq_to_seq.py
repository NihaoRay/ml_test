import torch
from torch import nn
from d2l import torch as d2l
import seq_to_seq as seq2seq


# 训练
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""

    # 权重初始化
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)

    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = seq2seq.MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        # 训练损失总和，词元数量
        metric = d2l.Accumulator(2)
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)

            # 强制教学
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)

            # 损失函数的标量进行 ”反向传播“
            l.sum().backward()
            # 梯度裁剪，防止梯度爆炸
            d2l.grad_clipping(net, 1)
            optimizer.step()

            # 有效长度
            num_tokens = Y_valid_len.sum()

            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} 'f'tokens/sec on {str(device)}')


embed_size, num_hiddens, num_layers, dropout = 256, 256, 6, 0.2
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

# 加载训练数据
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

# 初始化编码器与解码器
encoder = seq2seq.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = seq2seq.Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

# 合成编码器与解码器到一个类中，也就是让他们存在于一个网络中
net = d2l.EncoderDecoder(encoder, decoder)

train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

# 将训练好的模型以各个参数数据字典的形式保存
torch.save(net.state_dict(), "seq_to_seq.net.param")

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = d2l.predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {d2l.bleu(translation, fra, k=2):.3f}')

# d2l.plt.show()