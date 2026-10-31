import cv2
import numpy as np
import matplotlib.pyplot as plt

settings = [
    (1, "image1.png", 21, 400),   
    (2, "image2.png", 5, 30),     
    (3, "image3.png", 3, 30),    
    (3, "image3.png", 5, 30), 
]

for n, path, k, max_area in settings:

    # load in grayscale and convert to binary using Otsu threshold
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # structuring element (ellipse)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # single operations and the two combined sequences
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se)
    open_close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, se)
    close_open = cv2.morphologyEx(closing, cv2.MORPH_OPEN, se)

    names = ["Binary (input)", "Opening", "Closing", "Open then Close", "Close then Open"]
    images = [binary, opening, closing, open_close, close_open]

    print(f"\nImage {n}  (SE = {k}x{k} ellipse)")
    print(f"{'Result':18s}{'white specks':>14s}{'black cracks':>14s}{'changed pixels %':>18s}")

    for name, img in zip(names, images):
        # count small white specks
        num, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        white_specks = int((stats[1:, cv2.CC_STAT_AREA] <= max_area).sum())

        # count small black cracks 
        num, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(img), connectivity=8)
        black_cracks = int((stats[1:, cv2.CC_STAT_AREA] <= max_area).sum())

        changed = np.mean(img != binary) * 100
        print(f"{name:18s}{white_specks:>14d}{black_cracks:>14d}{changed:>18.2f}")

    # show all results in one figure
    plt.figure(figsize=(15, 6))
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.title(names[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"image{n}_se{k}_compare.png")
    plt.show()

# SE shape comparison on the fingerprint.
gray = cv2.imread("image3.png", cv2.IMREAD_GRAYSCALE)
_, binary3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

shapes = [("Rectangle", cv2.MORPH_RECT), ("Ellipse", cv2.MORPH_ELLIPSE), ("Cross", cv2.MORPH_CROSS)]

print("\nSE shape comparison (image 3, 5x5, open then close)")
plt.figure(figsize=(12, 4))
for i, (name, shp) in enumerate(shapes):
    se = cv2.getStructuringElement(shp, (5, 5))
    out = cv2.morphologyEx(binary3, cv2.MORPH_OPEN, se)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, se)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    white_specks = int((stats[1:, cv2.CC_STAT_AREA] <= 30).sum())
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(out), connectivity=8)
    black_cracks = int((stats[1:, cv2.CC_STAT_AREA] <= 30).sum())
    changed = np.mean(out != binary3) * 100

    print(f"{name:10s} white specks = {white_specks}, black cracks = {black_cracks}, changed pixels = {changed:.2f}%")
    plt.subplot(1, 3, i + 1)
    plt.imshow(out, cmap="gray")
    plt.title(name)
    plt.axis("off")
plt.tight_layout()
plt.savefig("se_shape_compare.png")
plt.show()

print("Done")