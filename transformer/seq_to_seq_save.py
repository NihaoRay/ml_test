import torch
from torch import nn
from d2l import torch as d2l
import seq_to_seq as seq2seq

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()


# 加载训练数据
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)


# 初始化编码器与解码器
encoder = seq2seq.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = seq2seq.Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)


net = d2l.EncoderDecoder(encoder, decoder).to(device)

# 加载已经训练完成的模型
net.load_state_dict(torch.load('seq_to_seq.net.param'))

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = d2l.predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {d2l.bleu(translation, fra, k=2):.3f}')

print(str(device))