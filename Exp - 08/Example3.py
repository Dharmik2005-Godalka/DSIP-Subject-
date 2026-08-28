import cv2
import numpy as np
from matplotlib import pyplot as plt

source_path = 'source.jpg'
reference_path = 'reference.jpg'

source_image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

source_hist = cv2.calcHist([source_image], [0], None, [256], [0, 256])
reference_hist = cv2.calcHist([reference_image], [0], None, [256], [0, 256])

source_hist /= source_hist.sum()
reference_hist /= reference_hist.sum()

# Calculating CDF
source_cdf = source_hist.cumsum()
reference_cdf = reference_hist.cumsum()

# mapping source CDF to reference CDF
mapping = np.interp(source_cdf, reference_cdf, range(256))
matched_image = mapping[source_image]

# convert to uint8 data type
matched_image = matched_image.astype(np.uint8)

plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.title('source img.')
plt.imshow(source_image, cmap='gray')
plt.axis('off')

plt.subplot(132)
plt.title('ref. img.')
plt.imshow(reference_image, cmap='gray')
plt.axis('off')

plt.subplot(133)
plt.title('matched img.')
plt.imshow(matched_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
