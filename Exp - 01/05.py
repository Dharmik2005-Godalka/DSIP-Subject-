import numpy as np
import matplotlib.pyplot as plt

def continuous(time, amplitude, coefficient):
    return amplitude * np.exp(coefficient * time)

def discrete(samples, amplitude, coefficient):
    return amplitude * np.exp(coefficient * np.arange(samples))

time = np.linspace(0, 5, 1000)
samples = 20
amplitude = 2
coefficient = -0.5

continuous_exponential = continuous(time, amplitude, coefficient)
discrete_exponential = discrete(samples, amplitude, coefficient)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, continuous_exponential)
plt.title('continuous exponential signal')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.subplot(2, 1, 2)
plt.stem(discrete_exponential)
plt.title('discrete exponential signal')
plt.xlabel('sample')
plt.ylabel('amplitude')

plt.tight_layout()
plt.show()