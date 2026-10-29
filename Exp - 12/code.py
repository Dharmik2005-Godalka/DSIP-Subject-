import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('img.png', 0)

# convert to a binary img...
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# hit and miss.
foreground = np.array([[0, 0, 0],
                       [0, 1, 1],
                       [0, 1, 1]], dtype=np.int8)

background = np.array([[1, 1, 1],
                       [1, 0, 0],
                       [1, 0, 0]], dtype=np.int8)

kernel = foreground - background
print("Kernel:\n", kernel)

# apply the Hit or Miss transformation:
result = cv2.morphologyEx(binary, cv2.MORPH_HITMISS, kernel)
print("No. of matched pixels:", np.count_nonzero(result))

result_show = cv2.dilate(result, np.ones((9, 9), np.uint8))

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(result_show, cmap='gray')
plt.title('Hit or Miss Result')
plt.axis('off')

plt.show()
cv2.imwrite('result_image.png', result)
