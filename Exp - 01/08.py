import numpy as np
import matplotlib.pyplot as plt

def unit(x):
    return np.where(x >= 0, 1, 0)

axis = np.linspace(-10, 10, 1000)

output_signal = unit(axis) + unit(axis - 1) + 3 * unit(axis + 5)

plt.plot(axis, output_signal)
plt.title('y(t) = u(t) + u(t-1) + 3u(t+5)')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.ylim([-0.5, 5.5])
plt.grid(True)
plt.show()
