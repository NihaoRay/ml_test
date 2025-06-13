import torch
from torch import nn
from d2l import torch as d2l


T = 1000 # 总共产⽣1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
# d2l.plt.show()


tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

import re

text = "JGood is a handsome boy, he is cool, clever, and so on..."
re.sub('[^A-Za-z]+', ' ', text).strip().lower()
print (re.sub('[^A-Za-z]+', ' ', text).strip().lower())
