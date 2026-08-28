import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = 'source.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# histogram:
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

plt.figure(figsize=(8, 6))
plt.title('Histogram')
plt.xlabel('Pixel val.')
plt.ylabel('Freq.')
plt.plot(histogram)
plt.xlim([0, 256])
plt.grid(True)
plt.show()
