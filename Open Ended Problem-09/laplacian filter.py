import cv2
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

#laplacian sharpening...
laplacian = cv2.Laplacian(img, cv2.CV_64F)
sharpened_img = img - laplacian

cv2.imwrite('laplacian_img.png', sharpened_img)

plt.imshow(sharpened_img, cmap='gray')
plt.title('Laplacian sharpened img.')
plt.axis('off')
plt.show()
