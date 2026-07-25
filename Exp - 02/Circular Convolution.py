import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
h = np.array([2, 4, 6, 8, 10])

N = len(x) + len(h) - 1
X = np.fft.fft(x, N)
H = np.fft.fft(h, N)
y = np.fft.ifft(X * H)

plt.stem(y.real)
plt.title('Circular convolution')

plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid(True)
plt.show()
