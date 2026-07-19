import numpy as np
import matplotlib.pyplot as plt

def unit(x, val):
    result = np.zeros(len(x))
    result[x == val] = 1
    return result

n_axis = np.arange(-10, 11)

signal = unit(n_axis, 0) + unit(n_axis, 1) + 3 * unit(n_axis, -5)

plt.stem(n_axis, signal)
plt.title('y(t) = delta(t) + delta(t-1) + 3delta(t+5)')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.ylim([-0.5, 4.5])
plt.grid(True)
plt.show()
