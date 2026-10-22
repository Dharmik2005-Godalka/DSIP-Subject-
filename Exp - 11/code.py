import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("/content/CPMAD logo.jpg", 0)

#kernel...
kernel = np.ones((5, 5), np.uint8)

#Dilation...
dilation = cv2.dilate(image, kernel, iterations=1)

# Erosion:
erosion = cv2.erode(image, kernel, iterations=1)

#Opening 
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Closing (Dilation followed by Erosion):
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(15, 5))

plt.subplot(1, 5, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(dilation, cmap='gray')
plt.title("Dilated Image")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(erosion, cmap='gray')
plt.title("Eroded Image")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(opening, cmap='gray')
plt.title("Opening")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(closing, cmap='gray')
plt.title("Closing")
plt.axis("off")

plt.tight_layout()
plt.show()