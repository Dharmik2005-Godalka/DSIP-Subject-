import numpy as np
import matplotlib.pyplot as plt

fs = 16000
duration = 0.01
frequencies = [1000, 2000, 3000, 4000]

t = np.arange(0, duration, 1 / fs)

# Add all 4 sine waves...
combined_signal = sum(np.sin(2 * np.pi * f * t) for f in frequencies)

#single-sided amplitude spectrum
spectrum = np.fft.rfft(combined_signal)
freq_axis = np.fft.rfftfreq(len(combined_signal), d=1 / fs)
magnitude = np.abs(spectrum) / len(combined_signal)
magnitude[1:-1] *= 2

fig, axes = plt.subplots(2, 1, figsize=(11, 8))

axes[0].plot(t * 1000, combined_signal)
axes[0].set_title("Combined Signal - Time Domain")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

axes[1].plot(freq_axis / 1000, magnitude)
axes[1].set_title("Combined Signal - Frequency Domain")
axes[1].set_xlabel("Frequency (kHz)")
axes[1].set_ylabel("Magnitude")
axes[1].grid(True)
axes[1].set_xlim(0, fs / 2000)

plt.tight_layout()
plt.show()
