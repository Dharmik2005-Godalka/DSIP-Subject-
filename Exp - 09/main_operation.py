import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/content/Quarter.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gaussian_1d = cv2.getGaussianKernel(5, 1.5) 
gaussian_kernel = np.outer(gaussian_1d, gaussian_1d) 

print("Gaussian Kernel:")
print(gaussian_kernel)

gaussian_image = cv2.filter2D(image, -1, gaussian_kernel)

kernel_size = (5, 5)
average_kernel = np.ones(kernel_size, dtype=np.float32) / (kernel_size[0] * kernel_size[1])

print("\nAveraging Kernel:")
print(average_kernel)

average_image = cv2.filter2D(image, -1, average_kernel)
median_image = cv2.medianBlur(image, 5)

sharpening_kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])

sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

laplacian_kernel = np.array([[ 0, -1,  0],
                             [-1,  5, -1],
                             [ 0, -1,  0]], dtype=np.float32)

highpass_image = cv2.filter2D(blurred_image, -1, laplacian_kernel)

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

images = [image_rgb,
          to_rgb(gaussian_image),
          to_rgb(average_image),
          to_rgb(median_image),
          to_rgb(sharpened_image),
          to_rgb(highpass_image)]

titles = ["Original Image",
          "Gaussian Smoothing",
          "Averaging Filter",
          "Median Filter",
          "Sharpening (Center 9)",
          "High-Pass / Laplacian"]

plt.figure(figsize=(15, 9))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()