import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

Fs = 10000      # Sampling freq.
Fc = 3000       
N = 31          

b = firwin(N, Fc, window='hann', pass_zero='lowpass', fs=Fs)
print("Coefficients:", np.round(b, 4))

#freq. response
f, H = freqz(b, worN=2048, fs=Fs)
mag = 20 * np.log10(np.maximum(np.abs(H), 1e-10))

plt.plot(f, mag)
plt.title('FIR Low Pass Filter (fc = 3 kHz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.ylim(-100, 5)
plt.grid(True)
plt.show()
