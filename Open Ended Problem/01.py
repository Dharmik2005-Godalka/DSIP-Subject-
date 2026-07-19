import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-3, 7)
x = np.array([2, 3, 2, 4, 2, 3, 2, 3, 2, 3])

plt.figure(figsize=(8,4))

markerline, stemlines, baseline = plt.stem(n, x)

plt.title("1st Signal:")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.xticks(n)
plt.yticks([0,1,2,3,4])
plt.grid(True)
plt.show()
