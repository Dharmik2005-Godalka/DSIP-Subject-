import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, sosfreqz

song, fs = librosa.load("Original_Song.mp3", sr=None)
print("Sample rate:", fs, " Duration:", round(len(song) / fs, 2), "sec")

# All filter that needed are here...
def low_pass(x, fc, fs, order=6):
    sos = butter(order, fc / (fs / 2), btype='low', output='sos')
    return sosfiltfilt(sos, x)

def band_pass(x, f1, f2, fs, order=6):
    sos = butter(order, [f1 / (fs / 2), f2 / (fs / 2)], btype='band', output='sos')
    return sosfiltfilt(sos, x)

def band_stop(x, f1, f2, fs, order=6):
    sos = butter(order, [f1 / (fs / 2), f2 / (fs / 2)], btype='bandstop', output='sos')
    return sosfiltfilt(sos, x)

def save_mp3(x, name, fs):
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.95    
    sf.write(name, x, fs)                          

def show_spec(x, fs, title):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(x)), ref=np.max)
    plt.figure(figsize=(9, 3.5))
    librosa.display.specshow(D, sr=fs, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    print( )
    
# a. low pass filter part:
cutoffs = [500, 1500, 4000]

# frequency response of the three filters
plt.figure(figsize=(8, 4))
for fc in cutoffs:
    sos = butter(6, fc / (fs / 2), btype='low', output='sos')
    w, h = sosfreqz(sos, worN=4096, fs=fs)
    plt.semilogx(w, 20 * np.log10(np.abs(h) + 1e-9), label=str(fc) + " Hz")
plt.ylim(-80, 5)
plt.xlim(20, fs / 2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.title('Low-pass filter response')
plt.legend()
plt.grid(True)
plt.show()
print( )

show_spec(song, fs, 'Original song')

for fc in cutoffs:
    out = low_pass(song, fc, fs)
    save_mp3(out, "lowpass_" + str(fc) + "Hz.mp3", fs)
    show_spec(out, fs, 'Low-pass, cutoff = ' + str(fc) + ' Hz')

    centroid = librosa.feature.spectral_centroid(y=out, sr=fs).mean()
    energy_left = 100 * np.sum(out ** 2) / np.sum(song ** 2)
    print("fc =", fc, "Hz | centroid =", round(centroid), "Hz | energy kept =",
          round(energy_left, 2), "%")
    
f1, f2 = 300, 3400

vocals_kept   = band_pass(song, f1, f2, fs)
vocals_removed = band_stop(song, f1, f2, fs)

save_mp3(vocals_removed, "vocals_suppressed.mp3", fs)
save_mp3(vocals_kept, "background_suppressed.mp3", fs)

show_spec(vocals_removed, fs, 'Vocals suppressed (band-stop 300-3400 Hz)')
show_spec(vocals_kept, fs, 'Background suppressed (band-pass 300-3400 Hz)')