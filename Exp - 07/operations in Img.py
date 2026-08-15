import cv2
import numpy as np
import matplotlib.pyplot as plt

src_image = cv2.imread('5825038.jpg', cv2.IMREAD_GRAYSCALE)

#Img Negation:
negative_image = 255 - src_image

# Thresholding:
_, thresholded_image = cv2.threshold(src_image, 128, 255, cv2.THRESH_BINARY)

gamma = 2.0

# normalize img to range(0,1)
Img1 = src_image / 255.0

#gamma correction
gamma_corrected_img = np.power(Img1, 1/gamma)

# Convert back to range
gamma_corrected_image = np.uint8(gamma_corrected_img * 255)

plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.imshow(src_image, cmap='gray')
plt.title("original")
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(negative_image, cmap='gray')
plt.title("negative")
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(thresholded_image, cmap='gray')
plt.title("Threshold")
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(gamma_corrected_image, cmap='gray')
plt.title("gamma")
plt.axis('off')

plt.show()
