from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# load song
song = AudioSegment.from_mp3("song.mp3")
song = song.set_channels(1)
data = np.array(song.get_array_of_samples()).astype(np.float32)
data_norm = data / np.max(np.abs(data))

# different kernels to try
kernels = {
    "spread_echo": np.array([1, 0, 1, 0, 1], dtype=np.float32),
    "smoothing": np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32),
    "difference": np.array([1, -1], dtype=np.float32),
    "delay_echo": np.array([1, 0, 0, 0, 0.5], dtype=np.float32),
}

for name, kernel in kernels.items():

    output = convolve(data, kernel, mode='same')
    output = output / np.max(np.abs(output))

    output_int16 = (output * 32767).astype(np.int16)
    output_song = AudioSegment(
        output_int16.tobytes(),
        frame_rate=song.frame_rate,
        sample_width=2,
        channels=1
    )
    output_song.export(f"output_{name}.wav", format="wav")

    plt.figure(figsize=(12, 5))

    plt.subplot(2, 1, 1)
    plt.plot(data_norm, color='blue')
    plt.title("Original Audio Signal")

    plt.subplot(2, 1, 2)
    plt.plot(output, color='red')
    plt.title(f"Output with Kernel: {name}")

    plt.tight_layout()
    plt.show()

    print(f"Saved output_{name}.wav")