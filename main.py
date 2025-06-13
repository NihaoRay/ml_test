import matplotlib.pyplot as plt
import numpy
import numpy as np

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    return x * w + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


w_cor = numpy.arange(0.0, 4.0, 0.1)
b_cor = numpy.arange(-2.0, 2.1, 0.1)




w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 1):
    print('w=', w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pre_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pre_val, loss_val)
    print('MSE', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list)
plt.ylabel('loss')
plt.xlabel('w')
plt.show()