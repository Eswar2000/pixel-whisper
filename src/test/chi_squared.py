import cv2
import numpy as np
from math import pow

def chi_square_test(image_path):
    """
    Perform chi-square analysis on image LSBs.
    Returns the Chi-square statistic and p-value estimate.
    """

    # Read image as grayscale for simplicity
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    flat = img.flatten()

    # Count frequencies of pixel values
    freq = np.bincount(flat, minlength=256)

    # Chi-square computation on pairs (pixel values 2m and 2m+1)
    chi_sq = 0.0
    for i in range(0, 256, 2):
        o_even = freq[i]     # observed frequency of even pixel value
        o_odd = freq[i + 1]  # observed frequency of odd pixel value
        total = o_even + o_odd
        if total == 0:
            continue
        expected = total / 2

        # Chi-square contribution from this pair
        chi_sq += pow(o_even - expected, 2) / expected
        chi_sq += pow(o_odd - expected, 2) / expected

    return chi_sq

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
