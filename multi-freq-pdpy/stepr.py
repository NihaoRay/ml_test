import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import itertools




class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass


class modeltrain():
    def __init__(self):
        super().__init__()
        self.net_1 = model()
        self.initial_lr = 0.1

    def train(self):
        optimizer_1 = torch.optim.SGD(self.net_1.parameters(), lr=self.initial_lr, momentum=0.5)
        scheduler_1 = StepLR(optimizer_1, step_size=1, gamma=0.85)
        print("初始化的学习率：", optimizer_1.defaults['lr'])

        for epoch in range(1, 2):
            # train
            self.net_1.zero_grad()


            optimizer_1.step()
            scheduler_1.step()
            print("第%d个epoch的学习率：%f" % (epoch, optimizer_1.param_groups[0]['lr']))
        self.initial_lr = scheduler_1.get_last_lr()[0]
        print(f'initial_lr:{self.initial_lr}')


if __name__ == '__main__':
    netrain = modeltrain()
    for i in range(1, 40):
        netrain.train()