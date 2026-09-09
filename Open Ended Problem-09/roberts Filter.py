import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
img = np.float64(img)

#Roberts kernel:
kernel_x = np.array([[1, 0], [0, -1]])
kernel_y = np.array([[0, 1], [-1, 0]])

Gx = cv2.filter2D(img, -1, kernel_x)
Gy = cv2.filter2D(img, -1, kernel_y)

roberts_img = np.sqrt(Gx**2 + Gy**2)

cv2.imwrite('roberts_img.png', roberts_img)

plt.imshow(roberts_img, cmap='gray')
plt.title('Roberts edge detection')
plt.axis('off')
plt.show()
