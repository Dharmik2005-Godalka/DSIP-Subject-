import numpy as np
import matplotlib.pyplot as plt

def u(n):
    return np.where(n >= 0, 1, 0)

n = np.arange(-2, 11)
x = u(n) - u(n-3) - 5*u(n-7)

plt.figure(figsize=(7,4))
plt.stem(n, x)
plt.title("4th signal:")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.yticks(np.arange(-6, 7, 1))
plt.grid(True)
plt.show()

