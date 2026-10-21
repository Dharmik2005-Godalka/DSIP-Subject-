import numpy as np
import matplotlib.pyplot as plt

fs = 16000
duration = 0.01
frequencies = [1000, 2000, 3000, 4000]

t = np.arange(0, duration, 1 / fs)

fig, axes = plt.subplots(4, 1, figsize=(11, 10))

for ax, f in zip(axes, frequencies):
    x = np.sin(2 * np.pi * f * t)
    ax.stem(t * 1000, x, basefmt="k-")
    ax.set_title(f"Discrete Sine Wave - {f // 1000} kHz")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    
plt.tight_layout()
plt.show()
