import numpy as np
import matplotlib.pyplot as plt

def continuous(time, slope):
    ramp = np.zeros_like(time)
    ramp[time >= 0] = slope * time[time >= 0]
    return ramp

def discrete(samples, slope):
    ramp = np.zeros(samples)
    ramp[samples // 2:] = slope * np.arange(samples // 2, samples)
    return ramp

time = np.linspace(-5, 5, 1000)
samples = 20
slope = 2

continuous_ramp = continuous(time, slope)
discrete_ramp = discrete(samples, slope)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, continuous_ramp)
plt.title('continuous ramp signal')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.subplot(2, 1, 2)
plt.stem(discrete_ramp)
plt.title('discrete ramp signal')
plt.xlabel('sample')
plt.ylabel('amplitude')

plt.tight_layout()
plt.show()
