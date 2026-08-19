import cv2
import numpy as np
import matplotlib.pyplot as plt

#Histogram Equalization
img6 = cv2.imread("images/image6.png", cv2.IMREAD_GRAYSCALE)
out6 = cv2.equalizeHist(img6)

cv2.imwrite("output/image6_enhanced.png", out6)

plt.subplot(1,2,1)
plt.imshow(img6, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(out6, cmap='gray')
plt.title("Enhanced")
plt.axis('off')

plt.savefig("output/image6_compare.png")
plt.show()


img7 = cv2.imread("images/image7.png", cv2.IMREAD_GRAYSCALE)
out7 = cv2.equalizeHist(img7)

cv2.imwrite("output/image7_enhanced.png", out7)

plt.subplot(1,2,1)
plt.imshow(img7, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(out7, cmap='gray')
plt.title("Enhanced")
plt.axis('off')

plt.savefig("output/image7_compare.png")
plt.show()


# Gamma Correction - 0.6
img8 = cv2.imread("images/image8.png", cv2.IMREAD_GRAYSCALE)

img8_float = img8.astype(np.float64) / 255.0
out8 = np.power(img8_float, 0.6) * 255
out8 = np.uint8(np.clip(out8, 0, 255))

cv2.imwrite("output/image8_enhanced.png", out8)

plt.subplot(1,2,1)
plt.imshow(img8, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(out8, cmap='gray')
plt.title("Enhanced")
plt.axis('off')

plt.savefig("output/image8_compare.png")
plt.show()


#Gamma Correction - 0.8
img9 = cv2.imread("images/image9.png", cv2.IMREAD_GRAYSCALE)

img9_float = img9.astype(np.float64) / 255.0
out9 = np.power(img9_float, 0.8) * 255
out9 = np.uint8(np.clip(out9, 0, 255))

cv2.imwrite("output/image9_enhanced.png", out9)

plt.subplot(1,2,1)
plt.imshow(img9, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(out9, cmap='gray')
plt.title("Enhanced")
plt.axis('off')

plt.savefig("output/image9_compare.png")
plt.show()


# Image10: Gamma Correction - 1.5
img10 = cv2.imread("images/image10.png", cv2.IMREAD_GRAYSCALE)

img10_float = img10.astype(np.float64) / 255.0
out10 = np.power(img10_float, 1.5) * 255
out10 = np.uint8(np.clip(out10, 0, 255))

cv2.imwrite("output/image10_enhanced.png", out10)

plt.subplot(1,2,1)
plt.imshow(img10, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(out10, cmap='gray')
plt.title("Enhanced")
plt.axis('off')

plt.savefig("output/image10_compare.png")
plt.show()

print("Done")
