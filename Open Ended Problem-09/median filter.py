import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

#median filter...
median_img = cv2.medianBlur(img, 5)

cv2.imwrite('median_img.png', median_img)

plt.imshow(median_img, cmap='gray')
plt.title('Median filtered imag.')
plt.axis('off')
plt.show()
