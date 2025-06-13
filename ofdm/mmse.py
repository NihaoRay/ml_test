import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from pyphysim.util.conversion import dB2Linear
from pyphysim.util.misc import randn_c


# 此函数基于输入数组生成矩阵，偏移量offset基于输入数组
def generate_square_matrix(arr_data, size, data_offset, datatype):
    aMatrix = np.mat(np.zeros(shape=(size, size))).astype(datatype)
    for i1 in range(size):
        for i2 in range(size):
            try:
                arr_index = i2 + data_offset - i1
                if arr_index < 0:
                    continue
                aMatrix[i2, i1] = arr_data[arr_index]
            except:
                break
    return aMatrix


def MMSE_equalizer(x, y, size, data_type):
    # 计算自相关性
    ryy = np.correlate(y, y, "full")
    # 计算互相关性
    rxy = np.correlate(x, y, "full")

    # 从 ryy 和向量 Rxy 形式 rxy 生成矩阵 Ryy
    Ryy = generate_square_matrix(ryy, size, ryy.argmax(), data_type)
    Rxy = np.mat(np.zeros(shape=(size, 1))).astype(data_type)
    # 计算偏移量
    offset = rxy.argmax() - (size >> 1)
    for i in range(size): Rxy[i, 0] = rxy[i + offset]
    MMSE_C_Vec = np.asarray(inv(Ryy) * Rxy).flatten()
    result = np.convolve(y, MMSE_C_Vec)
    leftside = size >> 1
    print("发送的signal: ", x)
    print("接收的signal:", y)
    print("恢复的signal:", result[leftside:leftside + x.size])
    return result[leftside:leftside + x.size]


x = np.array([0.73 + 0.59j, 0.43 + 1.01j, 0.41 + 0.3j, 1.24 + 1.1j, 0.55 + 0.83j])
SNR_dB = 30
snr_linear = dB2Linear(SNR_dB)
noise_power = 1 / snr_linear
# 噪声
n = np.math.sqrt(noise_power) * randn_c(x.size)
# 信道响应
h = randn_c(x.size)
#
y_z = h * x + n
# MMSE均衡
y_z /= h
print(y_z)
# 滤波器长度
filter_length = 1
z = MMSE_equalizer(x, y_z, filter_length, complex)

plt.plot(abs(x))
plt.plot(abs(z))
plt.grid(True)
plt.show()
plt.savefig('signal2.png')
