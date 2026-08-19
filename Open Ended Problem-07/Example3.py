import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
floatimg = img.astype(np.float64) / 255.0 

gamma = 0.25
gamma_img = np.power(floatimg, gamma)
gamma_img = np.uint8(np.clip(gamma_img * 255, 0, 255))
cv2.imwrite('gamma_img.png', gamma_img)

plt.imshow(gamma_img, cmap='gray')
plt.title('Gamma Corrected Img')
plt.axis('off')
plt.show()
