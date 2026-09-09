import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
img = np.float64(img)

Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

sobel_img = np.sqrt(Gx**2 + Gy**2)

cv2.imwrite('sobel_img.png', sobel_img)

plt.imshow(sobel_img, cmap='gray')
plt.title('Sobel edge detection')
plt.axis('off')
plt.show()
