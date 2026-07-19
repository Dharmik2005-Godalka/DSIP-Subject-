import numpy as np
import matplotlib.pyplot as plt

def delta(n):
    return np.where(n == 0, 1, 0)

n = np.arange(-3, 4)
x = delta(n) + 3*delta(n-1) + 5*delta(n+1)

plt.figure(figsize=(7,4))
plt.stem(n, x)
plt.title("5th signal:")
plt.xlabel("n")
plt.ylabel("x[n]")

plt.xticks(np.arange(-3, 4, 1))
plt.yticks(np.arange(0, 6, 1))
plt.grid(True)
plt.show()
