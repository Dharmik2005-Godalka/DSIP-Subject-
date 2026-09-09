import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
img = np.float64(img)

# prewitt kernel:
kernel_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]])

kernel_y = np.array([[-1, -1, -1],
                      [ 0,  0,  0],
                      [ 1,  1,  1]])

Gx = cv2.filter2D(img, -1, kernel_x)
Gy = cv2.filter2D(img, -1, kernel_y)

prewitt_img = np.sqrt(Gx**2 + Gy**2)

cv2.imwrite('prewitt_img.png', prewitt_img)

plt.imshow(prewitt_img, cmap='gray')
plt.title('Prewitt edge detection')
plt.axis('off')
plt.show()
