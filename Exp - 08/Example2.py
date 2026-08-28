import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = 'source.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# histogram equalization:
equalized_image = cv2.equalizeHist(image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')
plt.show()

histogram2 = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

plt.figure(figsize=(8, 6))
plt.title('Histogram after Equalization')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.plot(histogram2)
plt.xlim([0, 256])
plt.grid(True)
plt.show()
