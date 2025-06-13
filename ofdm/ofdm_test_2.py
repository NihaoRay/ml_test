import numpy as np
import scipy

import matplotlib.pyplot as plt
import matplotlib
# matplotlib inline
from matplotlib import animation
from IPython.display import HTML
from scipy.interpolate import interp1d

#####################################################
# author: chenrui
# 实现出OFDM从发送端至接收端的过程，除了OFDM的基本星座图映射，FFT，串并转换，bit转symbols过程，信噪比（SNR）计算
# 还涉及到了导频应用方面的信道评估与均衡，多径效应信道，
# 暂时没有实现信号同步的问题。
#####################################################

K = 64 # number of OFDM subcarriers
CP = K//4  # length of the cyclic prefix: 25% of the block
P = 8 # number of pilot carriers per OFDM block
pilotValue = 3+3j # The known value each pilot transmits

allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.

# For convenience of channel estimation, let's make the last carriers also be a pilot
pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
P = P+1

# data carriers are all remaining carriers
dataCarriers = np.delete(allCarriers, pilotCarriers)

print ("allCarriers:   %s" % allCarriers)
print ("pilotCarriers: %s" % pilotCarriers)
print ("dataCarriers:  %s" % dataCarriers)

#绘制载波与导频所占位置的图像
# plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
# plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')
# plt.grid()
# plt.show()


mu = 4 # bits per symbol (i.e. 16QAM)
payloadBits_per_OFDM = len(dataCarriers)*mu  # number of payload bits per OFDM symbol

mapping_table = {
    (0,0,0,0) : -3-3j,
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}

for b3 in [0, 1]:
    for b2 in [0, 1]:
        for b1 in [0, 1]:
            for b0 in [0, 1]:
                B = (b3, b2, b1, b0)
                Q = mapping_table[B]
                # plt.plot(Q.real, Q.imag, 'bo')
                # plt.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha='center')

# plt.grid()
# plt.show()

#inverse mapping of the mapping table
demapping_table = {v : k for k, v in mapping_table.items()}

channelResponse = np.array([1, 0, 0.3+0.3j])  # the impulse response of the wireless channel
H_exact = np.fft.fft(channelResponse, K)
# plt.plot(allCarriers, abs(H_exact))

SNRdb = 25  # signal to noise-ratio in dB at the receiver


bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
print ("Bits count: ", len(bits))
print ("First 20 bits: ", bits[:20])
print ("Mean of bits (should be around 0.5): ", np.mean(bits))

# 序列转为并行
def SP(bits):
    return bits.reshape(len(dataCarriers), mu)

bits_SP = SP(bits)
print("First 5 bit goups")
print(bits_SP[:5, :])

# 星座图映射
def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])

QAM = Mapping(bits_SP)
print ("First 5 QAM symbols and bits:")
print (bits_SP[:5,:])
print ("星座图映射后的前五个QAM码元(symbols): %s" % QAM[:5])

def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
    symbol[dataCarriers] = QAM_payload  # allocate the pilot subcarriers
    return symbol

OFDM_data = OFDM_symbol(QAM)
print ("OFDM数据与导频，OFDM_data: %s" % OFDM_data)
print ("Number of OFDM carriers in frequency domain: ", len(OFDM_data))

def IDFT(OFDM_block_data):
    return np.fft.ifft(OFDM_block_data)

# 发送前进行傅里叶逆变换
OFDM_time = IDFT(OFDM_data)
print ("Number of OFDM samples in time-domain before CP: ", len(OFDM_time))
print(OFDM_time)
# plt.plot(allCarriers, abs(OFDM_time))
# plt.show()

def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

OFDM_withCP = addCP(OFDM_time)
print ("Number of OFDM samples in time domain with CP: ", len(OFDM_withCP))
print("OFDM_time with CP: %s" % OFDM_withCP)

def channel(signal):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved ** 2))
    sigma2 = signal_power * 10**(-SNRdb/10)  # calculate noise power based on signal power and SNR
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

OFDM_TX = OFDM_withCP
OFDM_RX = channel(OFDM_TX)
# plt.figure(figsize=(8,2))
# plt.plot(abs(OFDM_TX), label='TX signal')
# plt.plot(abs(OFDM_RX), label='RX signal')
# plt.legend(fontsize=10)
# plt.xlabel('Time'); plt.ylabel('$|x(t)|$')
# plt.grid(True)
# plt.show()

# 去除CP(循环前缀)
def removeCP(signal):
    return signal[CP:(CP+K)]
OFDM_RX_noCP = removeCP(OFDM_RX)
print("remove CP: %s" % OFDM_RX_noCP)

# 对接收到的数据进行傅里叶变换，相当于解调
def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)
OFDM_demod = DFT(OFDM_RX_noCP)


# 信道估计
def channelEstimate(OFDM_demod):
    pilots = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal
    Hest_at_pilots = pilots / pilotValue # divide by the transmitted pilot values

    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase separately
    Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)

    plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
    plt.stem(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
    plt.plot(allCarriers, abs(Hest), label='Estimated channel via interpolation')
    plt.grid(True)
    plt.xlabel('Carrier index')
    plt.ylabel('$|H(f)|$')
    plt.legend(fontsize=10)
    plt.ylim(0, 2)
    plt.show()

    return Hest
Hest = channelEstimate(OFDM_demod)

# 信道均衡
def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest
equalized_Hest = equalize(OFDM_demod, Hest)

# 获取星座图中的symbols
def get_payload(equalized):
    return equalized[dataCarriers]
QAM_est = get_payload(equalized_Hest)
# plt.plot(QAM_est.real, QAM_est.imag, 'bo')
# plt.grid()
# plt.show()

# 将星座图中的symbols映射成比特位形式的码元，如3-3j附近的复数转为(1,0,0,0)码元
def Demapping(QAM):
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])
    # calculate distance of each RX point to each possible point
    dists = abs(QAM.reshape((-1, 1)) - constellation.reshape((1, -1)))
    # for each element in QAM, choose the index in constellation
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)
    # get back the real constellation point
    hardDecision = constellation[const_index]
    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision


PS_est, hardDecision = Demapping(QAM_est)
for qam, hard in zip(QAM_est, hardDecision):
    plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o')
    plt.plot(hardDecision.real, hardDecision.imag, 'ro')
plt.grid()
plt.show()

# 并行转为串行
def PS(bits):
    return bits.reshape((-1,))
bits_est = PS(PS_est)

print('bits_est: %s' % bits_est)
print ("Obtained Bit error rate: ", np.sum(abs(bits-bits_est))/len(bits))








