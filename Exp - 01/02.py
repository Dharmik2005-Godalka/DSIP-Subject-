import numpy as np
import matplotlib.pyplot as plt

def impulse(signal_length, period):
    impulse1 = np.zeros(signal_length)
    for n in range(signal_length):
        if n % period == 0:
            impulse1[n] = 1
    return impulse1

signal_length = 100
period = 10

impulse_signal = impulse(signal_length, period)

plt.stem(impulse_signal)
plt.title('Impulse Train')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()
