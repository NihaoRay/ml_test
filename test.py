# import matplotlib.pyplot as plt
# import numpy as np
# # from scipy.integrate import odeint
#
#
# import matplotlib.pyplot as plt
#
# plt.style.use('seaborn-whitegrid')
# import pandas as pd
#
# adult = pd.read_csv("adult_with_pii.csv")
#
#
# def laplace_mech(v, sensitivity, epsilon):
#     return v + np.random.laplace(loc=0, scale=sensitivity / epsilon)
#
#
# def pct_error(orig, priv):
#     return np.abs(orig - priv) / orig * 100.0
#
#
# def rand_resp_sales(response):
#     truthful_response = response == 'Sales'
#
#     # 第一次抛硬币
#     if np.random.randint(0, 2) == 0:
#         # 如果抛正面回答正确答案
#         return truthful_response
#     else:
#         # 第二次抛硬币
#         return np.random.randint(0, 2) == 0

# from binaryninja import *
# import time
#
# bv = BinaryViewType.get_view_of_file('util.o')
# bv.update_analysis()
# time.sleep(5)
# for func in bv.functions:
#     print(func.symbol.name)


# class Animal(object):  #  python3中所有类都可以继承于object基类
#     name = 3
#     age = 4
#     sex = 1
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#
#     def call(self):
#         print(self.name, '会叫')
#
# ######
# # 现在我们需要定义一个Cat 猫类继承于Animal，猫类比动物类多一个sex属性。
# ######
# class Cat(Animal):
#    def __init__(self, name, age, sex):
#        super(Cat, self).__init__(name,age)  # 不要忘记从Animal类引入属性
#        self.sex=sex
#
# if __name__ == '__main__':  # 单模块被引用时下面代码不会受影响，用于调试
#    c = Cat()  #  Cat继承了父类Animal的属性
#    c.call()  # 输出 喵喵 会叫 ，Cat继承了父类Animal的方法


# import torch
#
# tensor_0 = torch.randint(1, 10, size=(1, 3, 4))
# print(tensor_0)
#
# shape = tensor_0.reshape(tensor_0.shape[0], tensor_0.shape[1], 2, -1)
# print(shape)
#
# print("head is first:")
#
# print(shape.permute(0, 2, 1, 3))

# def Lorenz(y0, t, param):
#     p, r, b = param
#     x, y, z = y0
#     dx = -p*(x - y)
#     dy = r * x - y - x * z
#     dz = -b * z + x * y
#     return np.array([dx, dy, dz])
#
#
# t = np.arange(0, 50, 0.01)
# np.random.seed(0)
# y0 = np.random.randn(3)

# import numpy as np
# import cupy as np
#
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[11, 12], [13, 14]])
# print(np.dot(a, b))


# import argparse
#
# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
# parser.add_argument()


# plt.legend(['x', 'y', 'z'])
# plt.xlabel('time')
# plt.title(f'Lorenz r=' + str(30))
# sol= odeint(Lorenz, y0, t, args=([10, 30, 8/3], ))
#
# plt.axes(projection = '3d')
# plt.plot(sol[:,0], sol[:, 1], sol[:,2])
# plt.show()


#
# y, x = np.mgrid[-2:2:50j, -2:2:50j]
# u = x ** 3 + x * y ** 2 - 2 * x ** 2 - 3 * x - y
# v = y * x ** 2 + y ** 3 - 2 * x * y - 3 * y + x
#
# figure = plt.figure()
# ax = figure.add_subplot(111)
#
# ax.streamplot(x, y, u, v)
# plt.show()


# for i in [0, 10, 20, 30]:
#     plt.legend(['x', 'y', 'z'])
#     plt.xlabel('time')
#     plt.title(f'Lorenz r=' + str(i))
#     sol= odeint(Lorenz, y0, t, args=([10, i, 8/3], ))
#
#     plt.axes(projection = '3d')
#     plt.plot(sol[:,0], sol[:, 1], sol[:,2])
#     plt.show()


# a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9],
#                   [1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
#
# b = torch.tensor([44, 55, 66, 44, 55, 66, 44, 55, 66, 44, 55, 66])
#
# train_ids = data.TensorDataset(a, b) #相当于zip函数

# for x_train, y_label in train_ids:
#     print(x_train, y_label)

# print('=' * 30)

# train_loader = data.DataLoader(dataset=train_ids, batch_size=4, shuffle=True)  #shuffle参数：打乱数据顺序
# for i, data in enumerate(train_loader, 1):  # 注意enumerate返回值有两个,一个是序号，一个是数据（包含训练数据和标签）,参数1是设置从1开始编号
#     x_data, label = data
#     print(' batch:{0} x_data:{1}  label: {2}'.format(i, x_data, label))


# print(torch.__version__)


# matrix = torch.tensor([[1, 0, 0, 0], [-1, 1, 0, 0], [2, 0.5, 1, 0], [-1, 0.5, -1/3, 1]])
# print(torch.inverse(matrix))
# print(torch.mm(torch.inverse(matrix), matrix))


# x = torch.arange(1, 4)
# print(x)
#
# result = x / (x + x)
# print(result)


#
# torch.save(x, 'x-file')
#
#
# x2 = torch.load('x-file')
#
# print(x2)


# # 正则表达式的应用
# import re
#
# path = 'D:/localworkspace/article.txt'
# with open(path, 'r', encoding='UTF-8') as f:
#     article = f.readlines()
#     for line in article:
#         result = re.findall('{b\d+}', line)
#         if len(result) > 0:
#             for name in result:
#                 print(name)
