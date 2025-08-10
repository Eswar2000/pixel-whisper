import cv2
import numpy as np
from math import pow
from scipy.stats import skew, kurtosis


def chi_square_test_color(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    chi_sq_channels = {}
    for i, channel_name in enumerate(['Blue', 'Green', 'Red']):
        channel = img[:,:,i].flatten()
        freq = np.bincount(channel, minlength=256)

        chi_sq = 0.0
        for val in range(0, 256, 2):
            o_even = freq[val]
            o_odd = freq[val+1]
            total = o_even + o_odd
            if total == 0:
                continue
            expected = total / 2
            chi_sq += (o_even - expected)**2 / expected
            chi_sq += (o_odd - expected)**2 / expected

        chi_sq_channels[channel_name] = chi_sq

    return chi_sq_channels

def compare_chi_square_channels(chi_sq_1, chi_sq_2):
    """
    Compare chi-square values from two images channel-wise.
    Returns a dict with easy-to-print comparison summary.
    """
    epsilon = 1e-5
    comparison_dict = {}
    
    for ch in ['Blue', 'Green', 'Red']:
        val1 = chi_sq_1.get(ch, None)
        val2 = chi_sq_2.get(ch, None)
        if val1 is None or val2 is None:
            comparison_dict[ch] = {
                "image1": None,
                "image2": None,
                "status": "N/A (missing value)",
                "difference": None
            }
            continue
        
        diff = val2 - val1
        abs_diff = abs(diff)
        
        if abs_diff < epsilon:
            status = "equal"
            abs_diff = 0.0
        elif diff < 0:
            status = "less detectable"
        else:
            status = "more detectable"
        
        comparison_dict[ch] = {
            "image1": val1,
            "image2": val2,
            "status": status,
            "difference": abs_diff
        }
    
    return comparison_dict

def run_chi_square_test(cover_image_path, stego_image_path):
    chi_sq_cover_image = chi_square_test_color(cover_image_path)
    chi_sq_stego_image = chi_square_test_color(stego_image_path)
    return compare_chi_square_channels(chi_sq_cover_image, chi_sq_stego_image)

def normalized_cross_correlation(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    numerator = np.sum((img1 - mean1) * (img2 - mean2))
    denominator = np.sqrt(np.sum((img1 - mean1)**2) * np.sum((img2 - mean2)**2))
    if denominator == 0:
        return 0
    return numerator / denominator

def estimate_noise(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    noise_std = lap.std()
    return noise_std

def noise_difference(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    noise1 = estimate_noise(img1)
    noise2 = estimate_noise(img2)
    return noise2 - noise1  # positive means stego is noisier

def histogram_bhattacharyya_distance(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    hist1 = cv2.calcHist([img1], [0], None, [256], [0,256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0,256])
    
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    
    dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return dist  # 0 means identical histograms

def edge_difference(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    sobel1 = cv2.Sobel(img1, cv2.CV_64F, 1, 1)
    sobel2 = cv2.Sobel(img2, cv2.CV_64F, 1, 1)
    
    diff = np.abs(sobel1 - sobel2)
    return np.mean(diff)  # average edge difference

def skewness_kurtosis_difference(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE).flatten()
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE).flatten()

    skew_diff = abs(skew(img2) - skew(img1))
    kurt_diff = abs(kurtosis(img2) - kurtosis(img1))
    return skew_diff, kurt_diff