import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# negative transform
negative_img = 255 - img
cv2.imwrite('negative_img.png', negative_img)

# plot the result
plt.imshow(negative_img, cmap='gray')
plt.title('Negative Image')
plt.axis('off')
plt.show()
