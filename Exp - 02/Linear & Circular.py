import numpy as np
import matplotlib.pyplot as plt

def linear_convolution(signal1, signal2):
    linear = np.convolve(signal1, signal2, mode='full')
    return linear

def circular_convolution(signal1, signal2):
    fft_length = len(signal1) + len(signal2) - 1
    fft_signal1 = np.fft.fft(signal1, fft_length)
    fft_signal2 = np.fft.fft(signal2, fft_length)
    circular = np.fft.ifft(fft_signal1 * fft_signal2)
    return circular

signal1 = np.array([1, 2, 3, 4, 5])
signal2 = np.array([2, 4, 6, 8, 10])

linear_con = linear_convolution(signal1, signal2)
circular_con = circular_convolution(signal1, signal2)

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.stem(linear_con)
plt.title('Linear conv.')
plt.xlabel('sample')
plt.ylabel('amplitude')

plt.subplot(2, 1, 2)
plt.stem(circular_con.real)
plt.title('Circular conv.')
plt.xlabel('sample')
plt.ylabel('amplitude')

plt.tight_layout()
plt.show()
