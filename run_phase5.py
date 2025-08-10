import os
import sys
from src.test.chi_squared import chi_square_test_color, compare_chi_square_channels

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python runner_phase5.py <cover_image_path> <stego_image_path>")
        sys.exit(1)

    cover_image = sys.argv[1]
    stego_image = sys.argv[2]

    chi_sq_cover_image = chi_square_test_color(cover_image)
    chi_sq_stego_image = chi_square_test_color(stego_image)

    chi_sq_diff = compare_chi_square_channels(chi_sq_cover_image, chi_sq_stego_image)

    for ch, summary in chi_sq_diff.items():
        print(f"{ch}: {summary}")