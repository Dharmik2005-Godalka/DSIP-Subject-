import numpy as np
import matplotlib.pyplot as plt

signal = np.array([1,2,3,4,5,6,7,8])

fft_result = np.fft.fft(signal)
print(fft_result)

magnitude_spectrum = np.abs(fft_result)
print(magnitude_spectrum)

phase_spectrum = np.angle(fft_result)
print(phase_spectrum)

reconstructed_signal = np.fft.ifft(fft_result)
print(reconstructed_signal)

sample_index = np.arange(len(signal))

plt.figure(figsize=(12,10))

#plot-1
plt.subplot(2,2,1)
plt.stem(sample_index,signal)
plt.title('Original Signal')
plt.xlabel('Frequency Index')
plt.ylabel('Amplitude')
plt.grid(True)

#plot-2
plt.subplot(2,2,2)
plt.stem(sample_index,magnitude_spectrum)
plt.title('MAGNITUDE SIGNAL')
plt.xlabel('Frequency Index')
plt.ylabel('Magnitude Spectrum')
plt.grid(True)

#plot-3
plt.subplot(2,2,3)
plt.stem(sample_index,phase_spectrum)
plt.title('PHASE SIGNAL')
plt.xlabel('Frequency Index')
plt.ylabel('Phase Spectrum')
plt.grid(True)

#plot 4
plt.subplot(2,2,4)
plt.stem(sample_index,reconstructed_signal)
plt.title('RECONSTRUCTED SIGNAL')
plt.xlabel('Frequency Index')
plt.ylabel('Reconstructed Signal')
plt.grid(True)
