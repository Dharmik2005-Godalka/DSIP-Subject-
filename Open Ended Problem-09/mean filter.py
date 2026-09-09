import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

#mean filter:
mean_img = cv2.blur(img, (5,5))
cv2.imwrite('mean_img.png', mean_img)

plt.imshow(mean_img, cmap='gray')
plt.title('Mean filtered img.')
plt.axis('off')
plt.show()
