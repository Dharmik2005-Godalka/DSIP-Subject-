import numpy as np
import matplotlib.pyplot as plt

n = np.arange(0,12)
x = np.array([0,1,1,2,2,3,3,2,2,1,1,0])

plt.figure(figsize=(6,4))
plt.step(n, x, where='post')
plt.xlim(0, 12)
plt.ylim(0, 3.5)

plt.title("2nd signal:")
plt.xlabel("n")
plt.ylabel("x[n]")

plt.xticks(np.arange(0,13,1))
plt.yticks(np.arange(0,4,1))
plt.grid(True)
plt.show()
