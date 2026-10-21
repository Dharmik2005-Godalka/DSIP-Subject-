import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

fs = 16000
duration = 0.01
filter_order = 4
cutoff = 2000
frequencies = [1000, 2000, 3000, 4000]

t = np.arange(0, duration, 1 / fs)
combined_signal = sum(np.sin(2 * np.pi * f * t) for f in frequencies)

#apply Butterworth filter
b, a = butter(filter_order, cutoff, btype="low", fs=fs)
output = lfilter(b, a, combined_signal)

print("Numerator coefficients:", b)
print("Denominator coefficients:", a)

# Single-sided amplitude spectrum
spectrum = np.fft.rfft(output)
freq_axis = np.fft.rfftfreq(len(output), d=1 / fs)
magnitude = np.abs(spectrum) / len(output)
magnitude[1:-1] *= 2

fig, axes = plt.subplots(2, 1, figsize=(11, 8))

axes[0].plot(t * 1000, output)
axes[0].set_title("butterworth o/p - time domain (2 kHz cutoff)")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("amplitude")
axes[0].grid(True)

axes[1].plot(freq_axis / 1000, magnitude)
axes[1].set_title("butterworth o/p - freq. domain")
axes[1].set_xlabel("frequency (kHz)")
axes[1].set_ylabel("magnitude")
axes[1].grid(True)
axes[1].set_xlim(0, fs / 2000)

plt.tight_layout()
plt.show()
