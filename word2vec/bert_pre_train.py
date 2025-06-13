import torch
from torch import nn
from d2l import torch as d2l

batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)

torch.save(vocab, 'bert_pre_train.vacal')

# 在transformer中key_size, query_size, value_size必须要与num_hiddens保持一致。
# 因为transformer中的多头注意力q, k, v均是输入的x; 而输入的x是通过嵌入层（嵌入将一个字符id转换成num_hiddens长度矩阵）计算的。
# 因为有x与注意力计算后的值进行残差计算，所以注意力的结果向量大小与x、num_hiddens保持一致。

# mlm_in_features、hid_in_features的大小必须要与num_hiddens保持一致，因为mlm_in_features是接收最后的encoder的输出
net = d2l.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
            ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
            num_layers=2, dropout=0.2, key_size=128, query_size=128,
            value_size=128, hid_in_features=128, mlm_in_features=128,
            nsp_in_features=128)

devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()

def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):

    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])

    # 遮蔽语言模型损失的和，下⼀句预测任务损失的和，句子对的数量，计数
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y in train_iter:
            # 加载数据到gpu中
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])

            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = d2l._get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)

            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()

            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))

            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

        print(f'MLM loss {metric[0] / metric[3]:.3f}, '
              f'NSP loss {metric[1] / metric[3]:.3f}')
        print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
              f'{str(devices)}')

train_bert(train_iter, net, loss, len(vocab), devices, 50)

torch.save(net.state_dict(), 'bert_pre_train.pt')