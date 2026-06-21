import os
import csv
import re
import sys
import run_lsb_basic, run_chaotic_encrypt_lsb, run_chaos_dynamic_lsb, run_content_aware_lsb, run_chaos_compensation, run_lsb_compensation, run_meta_compensation, run_acme, run_carq, run_steganalysis_metrics

# Default secret payload used across all phases (override via CLI or main(message=...)).
DEFAULT_MESSAGE = "The quick brown fox jumps over the lazy dog"

# Regex to match all channels from chi-square output
CHI_PATTERN = re.compile(
    r"(Red|Green|Blue):\s*Image 1:\s*([-+]?\d*\.\d+|\d+),\s*Image 2:\s*([-+]?\d*\.\d+|\d+)\s*=>\s*(.+)"
)

# List of runner scripts for each phase
PHASE_RUNNERS = [
    ("phase1", run_lsb_basic.run_lsb_basic),
    ("phase2", run_chaotic_encrypt_lsb.run_chaotic_encrypt_lsb),
    ("phase3", run_chaos_dynamic_lsb.run_chaos_dynamic_lsb),
    ("phase4_1", run_content_aware_lsb.run_content_aware_lsb),
    ("phase4_2", run_chaos_compensation.run_chaos_compensation),
    ("phase4_3", run_lsb_compensation.run_lsb_compensation),
    ("phase4_4", run_meta_compensation.run_meta_compensation),
    ("phase4_5", run_acme.run_acme),
    ("carq", run_carq.run_carq),
]

# Paths
BASE_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(BASE_DIR, "images", "input")
SUMMARY_DIR = os.path.join(BASE_DIR, "summary")

def get_next_metrics_filename():
    # Pattern: metrics_<index>.csv
    pattern = re.compile(r"metrics_(\d+)\.csv$")
    max_idx = -1

    for fname in os.listdir(SUMMARY_DIR):
        match = pattern.match(fname)
        if match:
            idx = int(match.group(1))
            max_idx = max(max_idx, idx)

    next_idx = max_idx + 1
    return os.path.join(SUMMARY_DIR, f"metrics_{next_idx}.csv")

OUTPUT_CSV = get_next_metrics_filename()

def format_detectability(status, diff):
    if "Equal" in status or diff == 0 or diff == "" or diff is None:
        return status  # Just return status as is
    else:
        return f"{status} by {diff}"

def main(input_src = INPUT_DIR, message = DEFAULT_MESSAGE):
    # Prepare CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Cover Image", "Phase", "PSNR", "SSIM", "MSE", "Entropy Diff",
            "Blue_Cover", "Blue_Stego", "Blue_Detect",
            "Green_Cover", "Green_Stego", "Green_Detect",
            "Red_Cover", "Red_Stego", "Red_Detect",
            "NCC", "Noise Diff", "Histogram Dist", "Edge Diff", "Skewness Diff", "Kurtosis Diff"
        ])

        # Loop over all cover images
        for cover_image in os.listdir(input_src):
            if not cover_image.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            print(f"\n[*] Processing cover image: {cover_image}")
            cover_path = os.path.join(input_src, cover_image)

            for phase_runner in PHASE_RUNNERS:
                print(f"[*] Running {phase_runner[0]} on {cover_image}...")
                try:
                    metric_output = phase_runner[1](cover_path, message=message)
                except Exception as e:
                    print(f"[!] {phase_runner[0]} failed on {cover_image}: {e}")
                    continue

                phase_name = phase_runner[0]
                image_base_name = os.path.basename(cover_path).replace(".png", "").replace(".jpg", "").replace(".jpeg", "")
                stego_filename = f"{image_base_name}_{phase_name}.png"

                # Run chi-square for this cover and stego
                stego_path = os.path.join(BASE_DIR, "images", "output", stego_filename)
                test_metrics = run_steganalysis_metrics.run_all_metrics(cover_path, stego_path)

                # Extract PSNR, SSIM, MSE, entropy diff from embedding runner output
                psnr, ssim, mse, entropy_diff = round(metric_output['PSNR'], 4), round(metric_output['SSIM'], 4), round(metric_output['MSE'], 4), round(metric_output['Entropy Diff'], 4)

                writer.writerow([
                    cover_image, phase_runner[0],
                    psnr, ssim, mse, entropy_diff,
                    test_metrics["chi_square"]["Blue"].get("image1"), test_metrics["chi_square"]["Blue"].get("image2"), 
                        format_detectability(test_metrics["chi_square"]["Blue"].get("status", ""), test_metrics["chi_square"]["Blue"].get("difference", "")),
                    test_metrics["chi_square"]["Green"].get("image1"), test_metrics["chi_square"]["Green"].get("image2"), 
                        format_detectability(test_metrics["chi_square"]["Green"].get("status", ""), test_metrics["chi_square"]["Green"].get("difference", "")),
                    test_metrics["chi_square"]["Red"].get("image1"), test_metrics["chi_square"]["Red"].get("image2"), 
                        format_detectability(test_metrics["chi_square"]["Red"].get("status", ""), test_metrics["chi_square"]["Red"].get("difference", "")),
                    test_metrics["ncc"], test_metrics["noise_diff"], test_metrics["hist_dist"], test_metrics["edge_diff"], test_metrics["skew_diff"], test_metrics["kurt_diff"]
                ])

if __name__ == "__main__":
    # Optional args: <image_src_path> [secret_message]
    if len(sys.argv) < 2:
        print("Usage: python master_runner.py <image_src_path> [secret_message]")
        main()
    else:
        msg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MESSAGE
        main(sys.argv[1], message=msg)