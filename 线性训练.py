import matplotlib.pyplot as plt
import numpy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    return x * w + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

w_cor = numpy.arange(0.0, 4.0, 0.1)
b_cor = numpy.arange(-2.0, 2.1, 0.1)

# 此处直接使用矩阵进行计算
w, b = numpy.meshgrid(w_cor, b_cor)
mse = numpy.zeros(w.shape)

for x, y in zip(x_data, y_data):
    mse += loss(x, y)
mse = mse / 3

h = plt.contourf(w, b, mse)

fig = plt.figure()
ax = Axes3D(fig)
plt.xlabel(r'w', fontsize=20, color='cyan')
plt.ylabel(r'b', fontsize=20, color='cyan')
ax.plot_surface(w, b, mse, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.show()