import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from math import log2

def calculate_mse(img1, img2):
    """Mean Squared Error between two images."""
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def calculate_entropy(image):
    """Shannon entropy of the image."""
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256), density=True)
    histogram = histogram[histogram > 0]  # remove zero entries to avoid log(0)
    return -np.sum(histogram * np.log2(histogram))

def calculate_metrics(original, stego):
    """
    Returns a dictionary of metrics:
    - PSNR
    - SSIM
    - MSE
    - Entropy difference
    """

    psnr_val = psnr(original, stego, data_range=255)
    ssim_val = ssim(original, stego, channel_axis=-1)
    mse_val = calculate_mse(original, stego)
    entropy_original = calculate_entropy(original)
    entropy_stego = calculate_entropy(stego)
    entropy_diff = entropy_stego - entropy_original

    return {
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "MSE": mse_val,
        "Entropy Diff": entropy_diff
    }