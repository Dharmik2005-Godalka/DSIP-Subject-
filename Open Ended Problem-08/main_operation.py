import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("output", exist_ok=True)

# Histogram Matching fun...
def match_hist(img, mean, sigma):
    x = np.arange(256)
    target = np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    target_cdf = np.cumsum(target) / target.sum()

    hist = np.bincount(img.ravel(), minlength=256)
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0].min()
    cdf = (cdf - cdf_min) / (img.size - cdf_min)

    table = np.searchsorted(target_cdf, cdf - 1e-9).clip(0, 255)
    return np.uint8(table[img])

#histogram for each image.
targets = [(128, 50), (128, 50), (110, 45), (60, 45), (60, 45)]

for i in range(1, 6):
    img = cv2.imread("images/image" + str(i) + ".png", cv2.IMREAD_GRAYSCALE)

    he = cv2.equalizeHist(img)                             
    hm = match_hist(img, targets[i-1][0], targets[i-1][1])   

    cv2.imwrite("output/image" + str(i) + "_HE.png", he)
    cv2.imwrite("output/image" + str(i) + "_HM.png", hm)

    plt.subplot(2,3,1); plt.imshow(img, cmap='gray', vmin=0, vmax=255); plt.title("Original"); plt.axis('off')
    plt.subplot(2,3,2); plt.imshow(he, cmap='gray', vmin=0, vmax=255); plt.title("Equalization"); plt.axis('off')
    plt.subplot(2,3,3); plt.imshow(hm, cmap='gray', vmin=0, vmax=255); plt.title("Matching"); plt.axis('off')

    plt.subplot(2,3,4); plt.hist(img.ravel(), 256, range=(0,256), color='gray'); plt.yscale('log')
    plt.subplot(2,3,5); plt.hist(he.ravel(), 256, range=(0,256), color='gray'); plt.yscale('log')
    plt.subplot(2,3,6); plt.hist(hm.ravel(), 256, range=(0,256), color='gray'); plt.yscale('log')

    plt.savefig("output/image" + str(i) + "_compare.png")
    plt.show()

    print("Image", i)
    print("Original     : mean =", round(img.mean(),1), " std =", round(img.std(),1))
    print("Equalization : mean =", round(he.mean(),1), " std =", round(he.std(),1))
    print("Matching     : mean =", round(hm.mean(),1), " std =", round(hm.std(),1))

print("Done")