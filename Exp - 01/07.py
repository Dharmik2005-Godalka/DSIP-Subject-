import numpy as np
import matplotlib.pyplot as plt

def continuous(time, amplitude, frequency, phase):
    return amplitude * np.sin(2 * np.pi * frequency * time + phase)

def discrete(samples, sampling_frequency, amplitude, frequency, phase):
    time = np.arange(samples) / sampling_frequency
    return amplitude * np.sin(2 * np.pi * frequency * time + phase)

time = np.linspace(0, 1, 1000)
samples = 100
sampling_frequency = 10
amplitude = 1
frequency = 2
phase = 0

continuous_sine_wave = continuous(time, amplitude, frequency, phase)
discrete_sine_wave = discrete(samples, sampling_frequency, amplitude, frequency, phase)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, continuous_sine_wave)
plt.title('continuous sine wave signal')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.subplot(2, 1, 2)
plt.stem(discrete_sine_wave)
plt.title('discrete sine wave signal')
plt.xlabel('sample')
plt.ylabel('amplitude')

plt.tight_layout()
plt.show()
