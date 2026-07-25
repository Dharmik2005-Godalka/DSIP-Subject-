import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
h = np.array([2, 4, 6, 8, 10])

y = np.convolve(x, h)

plt.stem(y)
plt.title('Linear convolution')

plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid(True)
plt.show()
