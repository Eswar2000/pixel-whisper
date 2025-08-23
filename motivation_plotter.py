import cv2
import numpy as np
import matplotlib.pyplot as plt


IMAGE_PATH = "images/input/googleapi_160_200/img_179.jpg"
MASK_SAVE = "images/plot/perceptual_mask.png"
CAPACITY_SAVE = "images/plot/capacity_map.png"

## Load image
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## Generate Perceptual Mask (60% edges and 40% texture)
# Edge component
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
edges = np.sqrt(sobelx**2 + sobely**2)
edges = edges / edges.max()

# Texture component (variance in 5x5 neighborhood)
kernel_size = 5
mean = cv2.blur(gray.astype(np.float64), (kernel_size, kernel_size))
sqmean = cv2.blur((gray.astype(np.float64))**2, (kernel_size, kernel_size))
variance = sqmean - mean**2
variance = variance / variance.max()

# Weighted perceptual mask
perceptual_mask = 0.6 * edges + 0.4 * variance
perceptual_mask = perceptual_mask / perceptual_mask.max()

# Save visualization
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Cover Image")
ax = plt.subplot(1, 2, 2)
ax.set_facecolor('white') # bg color change to white
plt.imshow(perceptual_mask, cmap="coolwarm")
plt.axis("off")
plt.title("Perceptual Mask Heatmap")
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(MASK_SAVE, dpi=300)
plt.close()

## Generate Capacity Map (discretize mask into 0, 1 or 2 bits/pixel)
capacity_map = np.where(perceptual_mask > 0.6, 2,
                 np.where(perceptual_mask > 0.3, 1, 0))

# Save visualization
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Cover Image")
ax = plt.subplot(1, 2, 2)
ax.set_facecolor('white') # bg color change to white
plt.imshow(capacity_map, cmap="coolwarm")
plt.axis("off")
plt.title("Capacity Map (bits/pixel)")
plt.colorbar(fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(CAPACITY_SAVE, dpi=300)
plt.close()

## Report Statistics
total = capacity_map.size
cap0 = np.sum(capacity_map == 0)
cap1 = np.sum(capacity_map == 1)
cap2 = np.sum(capacity_map == 2)
print(f"Total pixels: {total}")
print(f"Cap=0: {cap0} ({cap0/total*100:.2f}%)")
print(f"Cap=1: {cap1} ({cap1/total*100:.2f}%)")
print(f"Cap=2: {cap2} ({cap2/total*100:.2f}%)")
print(f"Saved perceptual mask: {MASK_SAVE}")
print(f"Saved capacity map: {CAPACITY_SAVE}")