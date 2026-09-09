import cv2
import numpy as np

# image1, image2, image3: smoothing.

img1 = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('image3.png', cv2.IMREAD_GRAYSCALE)

# mean filter
mean1 = cv2.blur(img1, (5,5))
mean2 = cv2.blur(img2, (5,5))
mean3 = cv2.blur(img3, (5,5))

# median filter
median1 = cv2.medianBlur(img1, 5)
median2 = cv2.medianBlur(img2, 5)
median3 = cv2.medianBlur(img3, 5)

# gaussian filter
gauss1 = cv2.GaussianBlur(img1, (5,5), 1)
gauss2 = cv2.GaussianBlur(img2, (5,5), 1)
gauss3 = cv2.GaussianBlur(img3, (5,5), 1)

cv2.imwrite('image1_mean.png', mean1)
cv2.imwrite('image1_median.png', median1)
cv2.imwrite('image1_gauss.png', gauss1)

cv2.imwrite('image2_mean.png', mean2)
cv2.imwrite('image2_median.png', median2)
cv2.imwrite('image2_gauss.png', gauss2)

cv2.imwrite('image3_mean.png', mean3)
cv2.imwrite('image3_median.png', median3)
cv2.imwrite('image3_gauss.png', gauss3)


#image4, image5: sharpening.

img4 = cv2.imread('image4.png', cv2.IMREAD_GRAYSCALE)
img5 = cv2.imread('image5.png', cv2.IMREAD_GRAYSCALE)

# laplacian sharpening
lap4 = cv2.Laplacian(img4, cv2.CV_64F, ksize=3)
sharp4 = np.uint8(np.clip(img4 - lap4, 0, 255))

lap5 = cv2.Laplacian(img5, cv2.CV_64F, ksize=3)
sharp5 = np.uint8(np.clip(img5 - lap5, 0, 255))

cv2.imwrite('image4_sharp.png', sharp4)
cv2.imwrite('image5_sharp.png', sharp5)

print('done')
