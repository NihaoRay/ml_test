import os
import re
import torch
from torch import nn
from d2l import torch as d2l


d2l.DATA_HUB['SNLI'] = ('https://nlp.stanford.edu/projects/snli/snli_1.0.zip', '9fcde07509c7e87ec61c640c1b2753d9041758e4')
data_dir = "..\\data\\snli_1.0\\"

def read_snli(data_dir, is_train):
    """将SNLI数据集解析为前提、假设和标签"""
    def extract_text(s):
        # 删除我们不会使用的信息
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # 用⼀个空格替换两个或多个连续的空格
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()

    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt' if is_train else 'snli_1.0_test.txt')

    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]

    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]

    labels = [label_set[row[0]] for row in rows if row[0] in label_set]

    return premises, hypotheses, labels

train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('前提：', x0)
    print('假设：', x1)
    print('标签：', y)

test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])


train_iter, test_iter, vocab = d2l.load_data_snli(128, 50)


def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())

    if flatten:
        net.append(nn.Flatten(start_dim=1))

    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())

    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)

