import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image1:
img1 = cv2.imread("images/image1.png", cv2.IMREAD_GRAYSCALE)
out1 = cv2.equalizeHist(img1)
cv2.imwrite("output/image1_enhanced.png", out1)

plt.subplot(1,2,1);
plt.imshow(img1, cmap='gray'); 
plt.title("Original"); 
plt.axis('off')
plt.subplot(1,2,2); 
plt.imshow(out1, cmap='gray'); 
plt.title("Enhanced"); 
plt.axis('off')
plt.savefig("output/image1_compare.png")
plt.show()


#Image2:
img2 = cv2.imread("images/image2.png", cv2.IMREAD_GRAYSCALE)
out2 = cv2.equalizeHist(img2)
cv2.imwrite("output/image2_enhanced.png", out2)

plt.subplot(1,2,1); 
plt.imshow(img2, cmap='gray'); 
plt.title("Original"); 
plt.axis('off')
plt.subplot(1,2,2); 
plt.imshow(out2, cmap='gray'); 
plt.title("Enhanced"); 
plt.axis('off')
plt.savefig("output/image2_compare.png")
plt.show()


#image3: Gamma Correction- 0.4
img3 = cv2.imread("images/image3.png", cv2.IMREAD_GRAYSCALE)
img3_float = img3.astype(np.float64) / 255.0
out3 = np.power(img3_float, 0.4) * 255
out3 = np.uint8(np.clip(out3, 0, 255))
cv2.imwrite("output/image3_enhanced.png", out3)

plt.subplot(1,2,1); 
plt.imshow(img3, cmap='gray'); 
plt.title("Original"); 
plt.axis('off')
plt.subplot(1,2,2); 
plt.imshow(out3, cmap='gray'); 
plt.title("Enhanced"); 
plt.axis('off')
plt.savefig("output/image3_compare.png")
plt.show()


# image4: Gamma Correction- 0.25
img4 = cv2.imread("images/image4.png", cv2.IMREAD_GRAYSCALE)
img4_float = img4.astype(np.float64) / 255.0
out4 = np.power(img4_float, 0.25) * 255
out4 = np.uint8(np.clip(out4, 0, 255))
cv2.imwrite("output/image4_enhanced.png", out4)

plt.subplot(1,2,1); 
plt.imshow(img4, cmap='gray'); 
plt.title("Original"); 
plt.axis('off')
plt.subplot(1,2,2); 
plt.imshow(out4, cmap='gray'); 
plt.title("Enhanced"); 
plt.axis('off')
plt.savefig("output/image4_compare.png")
plt.show()


#image5: Gamma Correction- 0.25
img5 = cv2.imread("images/image5.png", cv2.IMREAD_GRAYSCALE)
img5_float = img5.astype(np.float64) / 255.0
out5 = np.power(img5_float, 0.25) * 255
out5 = np.uint8(np.clip(out5, 0, 255))
cv2.imwrite("output/image5_enhanced.png", out5)

plt.subplot(1,2,1); 
plt.imshow(img5, cmap='gray'); 
plt.title("Original"); 
plt.axis('off')
plt.subplot(1,2,2); plt.imshow(out5, cmap='gray'); 
plt.title("Enhanced"); 
plt.axis('off')
plt.savefig("output/image5_compare.png")
plt.show()

print("Done")
