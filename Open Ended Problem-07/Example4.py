import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
img_float = img.astype(np.float64)

r_min, r_max = img_float.min(), img_float.max()
stretched = (img_float - r_min) * 255 / (r_max - r_min)
stretched = np.uint8(np.clip(stretched, 0, 255))
cv2.imwrite('stretch_img.png', stretched)

plt.imshow(stretched, cmap='gray')
plt.title('Contrast Stretched Img.')
plt.axis('off')
plt.show()
