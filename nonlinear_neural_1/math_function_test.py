import numpy as np
import pandas as pd
import torch
from d2l import torch as d2l
from torch import nn
from torch.utils import data
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score
import datetime
import random
import matplotlib.pyplot as plt


class DataSet(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        return (self.features[index], self.labels[index])

    def __len__(self):
        return len(self.features)


def read_data():
    path = 'non_function.csv'
    df_chunk = pd.read_csv(path, chunksize=1e6, iterator=True, sep='\t', encoding='UTF-8')
    data = pd.concat([chunk for chunk in df_chunk])

    labels = data['y']
    samples = data.drop('y', axis=1)

    return samples, labels


class ClassFuntionsModel(nn.Module):
    def __init__(self, input_size, num_hiddens):
        super(ClassFuntionsModel, self).__init__()
        self.input_size = input_size
        self.num_hiddens = num_hiddens
        self.dense = nn.Sequential(nn.Linear(3, 32), nn.ReLU(),
                                   nn.Linear(32, 16), nn.ReLU(),
                                   nn.Linear(16, 8), nn.ReLU(),
                                   nn.Linear(8, 1))

    def forward(self, inputs):
        return self.dense(inputs)


def math_function(x1, x2, x3):
    return x1 ** 2 + 2 * x2 ** 3 + x3

def write_sample_to_csv():
    list = []
    y_list = []
    for i in range(100000):
        sample = []
        y_sample = []
        x1, x2, x3 = random.randint(0, 100) / 100, (random.randint(0, 100) - 1) / 100, (
                    random.randint(0, 100) - 2) / 100
        y = math_function(x1, x2, x3)

        sample.append(x1)
        sample.append(x2)
        sample.append(x3)

        y_sample.append(y)

        list.append(sample)
        y_list.append(y_sample)

    return torch.tensor(list, dtype=torch.float32), torch.tensor(y_list, dtype=torch.float32)


def sample_test():
    list = []
    y_list = []
    for i in range(26):
        sample = []
        y_sample = []
        x1, x2, x3 = random.randint(0, 100) / 100, (random.randint(0, 100) - 1) / 100, (
                random.randint(0, 100) - 2) / 100
        y = math_function(x1, x2, x3)

        sample.append(x1)
        sample.append(x2)
        sample.append(x3)

        y_sample.append(y)

        list.append(sample)
        y_list.append(y_sample)

    return torch.tensor(list, dtype=torch.float32), torch.tensor(y_list, dtype=torch.float32)


if __name__ == "__main__":
    # def main_function():
    batch_size, lr, num_epochs = 1500, 0.0015, 200
    net = ClassFuntionsModel(3, 16)

    updater = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.MSELoss()
    devices = d2l.try_all_gpus()
    x_train, y_train = write_sample_to_csv()

    # 加载训练数据集合
    train_dataset = DataSet(x_train, y_train)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                             num_workers=4)

    plt.ion()  # 实时打印的过程
    plt.show()
    x_test, y_test = sample_test()
    x_test = x_test.to(devices[0])
    y_test = y_test.to(devices[0])

    net.to(devices[0])
    for epoch in range(num_epochs):
        for i, (X, y) in enumerate(train_iter):
            X = X.to(devices[0])
            y = y.to(devices[0])
            y_hat = net(X)
            l = loss(y_hat, y)
            updater.zero_grad()
            l.backward()
            updater.step()

            if i % 8 == 0:
                print(f'loss {l:.5f}')

        input = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        input = input.to(devices[0])
        print(f'模型产生的数据：{net(input)}, 标准：{math_function(0.1, 0.2, 0.3)}')

        if epoch % 2 == 0:
            prediction = net(x_test)
            print(f'模型产生的数据：{prediction}, y_test:{y_test}')

            plt.cla()  # 清除当前图形中的当前活动轴。其他轴不受影响
            plt.scatter(np.arange(0, 26, 1), y_test.cpu().data.numpy())  # 打印原始数据
            plt.plot(np.arange(0, 26, 1), prediction.cpu().data.numpy(), 'r-', lw=5)  # 打印预测数据
            # plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})  # 打印误差值
            plt.pause(0.1)  # 每次停顿0.1



    plt.ioff()
    plt.show()