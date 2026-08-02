import librosa
import numpy as np
import matplotlib.pyplot as plt

# loading three songs
song1, fs = librosa.load("1st_Original_Song.mp3", sr=22050)
song2, fs = librosa.load("1st_SongMusic.mp3", sr=22050)
song3, fs = librosa.load("2nd_Song.mp3", sr=22050)

# keep only first 20 seconds:
song1 = song1[:fs * 20]
song2 = song2[:fs * 20]
song3 = song3[:fs * 20]

# makeing all three songs same length
min_len = min(len(song1), len(song2), len(song3))
song1 = song1[:min_len]
song2 = song2[:min_len]
song3 = song3[:min_len]


def check_match(a, b):
    # correlation between a and b
    result = np.correlate(a, b, mode='full')
    # divide by strength of a & strength of b
    result = result / (np.linalg.norm(a) * np.linalg.norm(b))
    return result


# shifting values:
shift = np.arange(-(len(song1) - 1), len(song1))

match_1_2 = check_match(song1, song2)   # 1st vs music
match_1_3 = check_match(song1, song3)   # 1st vs 2nd
match_2_3 = check_match(song2, song3)   # music vs 2nd

print("song1 vs song2 match :", round(max(match_1_2), 3))
print("song1 vs song3 match :", round(max(match_1_3), 3))
print("song2 vs song3 match :", round(max(match_2_3), 3))

plt.figure(figsize=(9, 7))

plt.subplot(3, 1, 1)
plt.plot(shift, match_1_2)
plt.title('Original song vs music ver.')
plt.xlabel('shift')
plt.ylabel('match value')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(shift, match_1_3)
plt.title('Orig. song vs diff. song')
plt.xlabel('shift')
plt.ylabel('match value')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(shift, match_2_3)
plt.title('Music ver. vs diff. song')
plt.xlabel('shift')
plt.ylabel('match value')
plt.grid(True)

plt.tight_layout()
plt.show()
