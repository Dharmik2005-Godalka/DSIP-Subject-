import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

Fs = 20000          
F1, F2 = 3000, 6000 # Lower and upper cutoff
N = 31              # No. of tap

#band stop filter
b = firwin(N, [F1, F2], window='hann', pass_zero='bandstop', fs=Fs)
print("Coefficients:", np.round(b, 4))

f, H = freqz(b, worN=2048, fs=Fs)
mag = 20 * np.log10(np.maximum(np.abs(H), 1e-10))

plt.plot(f, mag)
plt.title('FIR Band Stop Filter (3 kHz - 6 kHz)')
plt.xlabel('freq. (Hz)')
plt.ylabel('magnitude (dB)')
plt.ylim(-100, 5)
plt.grid(True)
plt.show()
