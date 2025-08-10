import os
import sys
from src.test.detector import run_chi_square_test, normalized_cross_correlation, noise_difference, histogram_bhattacharyya_distance, edge_difference, skewness_kurtosis_difference

def run_all_metrics(cover_path, stego_path):
    chi_sq_diff = run_chi_square_test(cover_path, stego_path)
    ncc = normalized_cross_correlation(cover_path, stego_path)
    noise_diff = noise_difference(cover_path, stego_path)
    hist_dist = histogram_bhattacharyya_distance(cover_path, stego_path)
    edge_diff_val = edge_difference(cover_path, stego_path)
    skew_diff, kurt_diff = skewness_kurtosis_difference(cover_path, stego_path)

    return {
        "chi_square": chi_sq_diff,
        "ncc": ncc,
        "noise_diff": noise_diff,
        "hist_dist": hist_dist,
        "edge_diff": edge_diff_val,
        "skew_diff": skew_diff,
        "kurt_diff": kurt_diff
    }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python runner_phase5.py <cover_image_path> <stego_image_path>")
        sys.exit(1)

    cover_image = sys.argv[1]
    stego_image = sys.argv[2]

    metrics = run_all_metrics(cover_image, stego_image)