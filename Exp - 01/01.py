import numpy as np
import matplotlib.pyplot as plt

def unit_impulse(length, position):
    signal = np.zeros(length)
    signal[position] = 1
    return signal

start, stop, step = -10, 10, 1
x = np.arange(start, stop + step, step)

signal1 = unit_impulse(len(x), abs(start)//step)

plt.stem(x, signal1)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title('unit impulse signal')
plt.grid(True)
plt.show()
