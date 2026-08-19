import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# histogram:
eq_img = cv2.equalizeHist(img)
cv2.imwrite('histeq_img.png', eq_img)

plt.imshow(eq_img, cmap='gray')
plt.title('Histogram Equalized Img.')
plt.axis('off')
plt.show()
