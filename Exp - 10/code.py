import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img.jpg", 0)

# fourier transformation
F = np.fft.fftshift(np.fft.fft2(img))
M, N = img.shape

# creating freq. distance matrix
u, v = np.meshgrid(np.arange(N)-N//2, np.arange(M)-M//2)
D = np.sqrt(u**2 + v**2)

#low pass and high Pass filter...
LPF = D < 30
HPF = D >= 30

# applying filters
smooth = np.abs(np.fft.ifft2(np.fft.ifftshift(F * LPF)))
high_pass = np.abs(np.fft.ifft2(np.fft.ifftshift(F * HPF)))
sharp = np.clip(img.astype(float) + high_pass, 0, 255)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('org. Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(smooth, cmap='gray')
plt.title('LPF - smoothening')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sharp, cmap='gray')
plt.title('HPF - sharpening')
plt.axis('off')
plt.show()