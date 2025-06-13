import serial
import time

####
# 电脑端与串口设备进行通信

#打开端口
serialPort = "COM6"
baudRate = 9600
ser = serial.Serial(serialPort, baudRate, timeout=0.5)

arduio_data = "test"
while True:
    time_start = time.time()
    arduio_data = ser.readline().decode('UTF-8')
    if arduio_data:
        print(arduio_data, end="")
        print(time.time() - time_start)

