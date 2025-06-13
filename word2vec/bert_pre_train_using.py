import torch
from d2l import torch as d2l

# 辞典需要使用训练过程中的辞典，每次辞典在加入数据集合的时候会产生些许的偏差，导致bert产生的结果不一样
batch_size, max_len = 512, 64
vocab = torch.load('bert_pre_train.vacal')
devices = d2l.try_all_gpus()


net = d2l.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
            ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
            num_layers=2, dropout=0.2, key_size=128, query_size=128,
            value_size=128, hid_in_features=128, mlm_in_features=128,
            nsp_in_features=128).to(devices[0])

# 屏蔽dropout与规范化的操作，如果net.train()是开启dropout与规范化的操作。
# dropout权重衰退机制有在“权重衰退.txt”有解释
net.eval()
net.load_state_dict(torch.load("bert_pre_train.pt"))


def get_bert_encoding(net, tokens_a, tokens_b = None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)

    return encoded_X

tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]

print(encoded_text, encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3])