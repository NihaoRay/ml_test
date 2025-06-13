# Bias-SVD代码实现
# 基于MovieLens 1M数据集
# 帮助文档的地址
# http://qzmvc1.top/%E6%88%91%E7%9A%84%E6%8E%A8%E8%8D%90%E7%AE%97%E6%B3%95%E4%B9%8B%E8%B7%AF-2-%E7%9F%A9%E9%98%B5%E5%88%86%E8%A7%A3.html

# from moviesData import readRatings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from d2l import torch as d2l

class MFDataset(Dataset):
    def __init__(self, u_id, i_id, rating):
        self.u_id = u_id
        self.i_id = i_id
        self.rating = rating

    def __getitem__(self, index):
        return self.u_id[index], self.i_id[index], self.rating[index]

    def __len__(self):
        return len(self.rating)


class MF(nn.Module):
    def __init__(self, num_users, num_items, mean, embedding_size=100):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)  # 隐向量
        self.user_bias = nn.Embedding(num_users, 1)  # 用户偏置
        self.item_emb = nn.Embedding(num_items, embedding_size)  # 隐向量
        self.item_bias = nn.Embedding(num_items, 1)  # 物品偏置

        # 参数初始化
        self.user_emb.weight.data.uniform_(0, 0.005)  # 0-0.05之间均匀分布
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        # 全局偏置
        # 将不可训练的tensor转换成可训练的类型parameter，并绑定到module里，net.parameter()中就有了这个参数
        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        # return (U * I).sum(1) + b_u + b_i + self.mean  # 返回预测评分
        return torch.diag(torch.matmul(U, I.permute(1, 0))) + b_u + b_i + self.mean  # 返回预测评分


def train(model, x_train, y_train, loss_func, device):
    train_ls = []  # 返回训练误差
    train_dataset = MFDataset(x_train[:, 0], x_train[:, 1], y_train)
    # DataLoader将一个batch_size封装成一个tensor，方便迭代
    train_iter = DataLoader(train_dataset, batch_size=1024)

    # weight_decay是正则化系数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)

    model = model.float()
    model = model.to(device)

    for epoch in range(100):
        model.train()  # 如果模型中有Batch Normalization或Dropout层，需要在训练时添加model.train()，使起作用
        total_loss, total_len = 0.0, 0
        for x_u, x_i, y in train_iter:
            x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            y_pred = model(x_u, x_i)
            l = loss_func(y_pred, y).sum()
            optimizer.zero_grad()  # 清空这一批的梯度
            l.backward()  # 回传
            optimizer.step()  # 参数更新

            total_loss += l.cpu().item()
            total_len += len(y)
        print(f"{epoch +  1}, train_ls: {total_loss / total_len}")
        train_ls.append(total_loss / total_len)
    return train_ls


def readMovies(path):
    movies = pd.read_table(os.path.join(path, 'movies.dat'), header=None, sep='::', engine='python')
    movies.columns = ['MovieID', 'Title', 'Genres']
    return movies

def readUsers(path):
    """ Age: {1, 'under 18',  18: "18-24",  25: "25-34",  35: "35-44",
              45: "45-49",    50: "50-55",  56: "56+",}"""
    users = pd.read_table(os.path.join(path, 'users.dat'), header=None, sep='::', engine='python')
    users.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
    return users

def readRatings(path):
    ratings = pd.read_table(os.path.join(path, 'ratings.dat'), header=None, sep='::', engine='python')
    ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    return ratings


def train_model():
    pd.set_option('display.max_rows', 1000, 'display.max_columns', None,
                  'display.float_format', lambda x: "%.2f" % x)
    path = r'D:\workspace\ml_dataset\recommand\ml-1m'
    df = readRatings(path)

    x, y = df.iloc[:, :2], df.iloc[:, 2]
    x = torch.tensor(x.values, dtype=torch.int64)
    y = torch.tensor(y.values, dtype=torch.float32)

    x_train, x_test, y_train, y_test = train_test_split(x.numpy(), y.numpy(), test_size=0.3, random_state=1)

    mean_rating = df.iloc[:, 2].mean()
    # 虽然数据集的UserID是从1开始的，但这里还是需要+1，因为nn.Embedding是从索引0开始，而model(x_u, x_i)传的是真实的ID
    num_users, num_items = df['UserID'].max() + 1, df['MovieID'].max() + 1

    device = d2l.try_gpu()

    model = MF(num_users, num_items, mean_rating)
    loss = nn.MSELoss(reduction='sum')

    train_ls = train(model, x_train, y_train, loss, device)
    print(train_ls)
    torch.save(model.state_dict(), 'basic-svd-recommand.pth')


def evalute_model():
    pd.set_option('display.max_rows', 1000, 'display.max_columns', None,
                  'display.float_format', lambda x: "%.2f" % x)
    path = r'D:\workspace\ml_dataset\recommand\ml-1m'
    df = readRatings(path)

    x, y = df.iloc[:, :2], df.iloc[:, 2]
    x = torch.tensor(x.values, dtype=torch.int64)
    y = torch.tensor(y.values, dtype=torch.float32)

    x_train, x_test, y_train, y_test = train_test_split(x.numpy(), y.numpy(), test_size=0.3, random_state=1)
    mean_rating = df.iloc[:, 2].mean()
    # 虽然数据集的UserID是从1开始的，但这里还是需要+1，因为nn.Embedding是从索引0开始，而model(x_u, x_i)传的是真实的ID
    num_users, num_items = df['UserID'].max() + 1, df['MovieID'].max() + 1

    device = d2l.try_gpu()
    model = MF(num_users, num_items, mean_rating).to(device)

    model.load_state_dict(torch.load("basic-svd-recommand.pth"))
    # 测试结果
    model.eval()

    print(model(torch.tensor(x_test[:100, 0]).to(device), torch.tensor(x_test[:100, 1]).to(device)))
    np.set_printoptions(threshold=np.inf)
    print(y_test[:100])
    print("结果")


if __name__ == '__main__':
    # train_model()
    evalute_model()


