import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

#gaussian filter... 
gaussian_img = cv2.GaussianBlur(img, (5,5), 0)

cv2.imwrite('gaussian_img.png', gaussian_img)

plt.imshow(gaussian_img, cmap='gray')
plt.title('Gaussian based img.')
plt.axis('off')
plt.show()
