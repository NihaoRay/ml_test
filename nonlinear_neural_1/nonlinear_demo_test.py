import torch
import torchvision
from IPython import display
from matplotlib import pyplot as plt
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

def aa_1():
    return 1, 2

a = aa_1()
b=12
# 要求把这两个元组合并成一个二维原则
c=a+(b,)
print(c)












