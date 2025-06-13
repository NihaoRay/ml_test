import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


data = pd.DataFrame({"学号":[1, 2, 3, 4],
                    "录取":["清华","北大","清华","蓝翔"],
                    "学历":["本科","本科","本科","专科"]})
data = pd.get_dummies(data)
print(data.shape)
print(data)

