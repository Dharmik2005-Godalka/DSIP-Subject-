import numpy as np
import matplotlib.pyplot as plt

# two signals
sig1 = np.array([1, 2, 3, 4, 5])
sig2 = np.array([2, 4, 6, 8, 10])

cross = np.correlate(sig1, sig2, mode='full')
auto = np.correlate(sig1, sig1, mode='full')

shift = np.arange(-(len(sig1) - 1), len(sig1))

# print all values
print("sig1 :", sig1)
print("sig2 :", sig2)
print("cross correlation :", cross)
print("auto correlation :", auto)

# cross correlation:
plt.subplot(2, 1, 1)
plt.stem(shift, cross)
plt.title('Cross correlation of sig1 and sig2')
plt.xlabel('shift')
plt.ylabel('amplitude')
plt.grid(True)

# auto correlation:
plt.subplot(2, 1, 2)
plt.stem(shift, auto)
plt.title('Auto correlation of signal1')
plt.xlabel('shift')
plt.ylabel('amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()
