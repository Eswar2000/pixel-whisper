import os
import csv
import subprocess
import re
import sys
import run_phase1, run_phase2, run_phase3, run_phase4_1, run_phase4_2, run_phase4_3, run_phase4_4

# Regex to match all channels from chi-square output
CHI_PATTERN = re.compile(
    r"(Red|Green|Blue):\s*Image 1:\s*([-+]?\d*\.\d+|\d+),\s*Image 2:\s*([-+]?\d*\.\d+|\d+)\s*=>\s*(.+)"
)

# List of runner scripts for each phase
PHASE_RUNNERS = [
    ("phase1", run_phase1.run_phase1),
    ("phase2", run_phase2.run_phase2),
    ("phase3", run_phase3.run_phase3),
    ("phase4_1", run_phase4_1.run_phase4_1),
    ("phase4_2", run_phase4_2.run_phase4_2),
    ("phase4_3", run_phase4_3.run_phase4_3),
    ("phase4_4", run_phase4_4.run_phase4_4)
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

def run_command(cmd):
    """Run shell command and return stdout."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] Command failed: {cmd}")
        print(result.stderr)
    return result.stdout.strip()

def main(input_src = INPUT_DIR):
    # Prepare CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Cover Image", "Phase", "PSNR", "SSIM", "MSE", "Entropy Diff",
            "Blue_Cover", "Blue_Stego", "Blue_Detect",
            "Green_Cover", "Green_Stego", "Green_Detect",
            "Red_Cover", "Red_Stego", "Red_Detect"
        ])

        # Loop over all cover images
        for cover_image in os.listdir(input_src):
            if not cover_image.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            print(f"\n[*] Processing cover image: {cover_image}")
            cover_path = os.path.join(input_src, cover_image)

            for phase_runner in PHASE_RUNNERS:
                print(f"[*] Running {phase_runner[0]} on {cover_image}...")
                metric_output = phase_runner[1](cover_path)

                phase_name = phase_runner[0]
                image_base_name = os.path.basename(cover_path).replace(".png", "").replace(".jpg", "").replace(".jpeg", "")
                stego_filename = f"{image_base_name}_{phase_name}.png"

                # Run chi-square for this cover and stego
                stego_path = os.path.join(BASE_DIR, "images", "output", stego_filename)
                chi_output = run_command(f"python run_phase5.py {cover_path} {stego_path}")

                # Extract PSNR, SSIM, MSE, entropy diff from embedding runner output
                psnr, ssim, mse, entropy_diff = round(metric_output['PSNR'], 4), round(metric_output['SSIM'], 4), round(metric_output['MSE'], 4), round(metric_output['Entropy Diff'], 4)

                # Parse chi-square output
                chi_data = {"Blue": {}, "Green": {}, "Red": {}}
                for line in chi_output.splitlines():
                    m = CHI_PATTERN.search(line)
                    if m:
                        color, img1_val, img2_val, detect_msg = m.groups()
                        chi_data[color] = {
                            "Image1": float(img1_val),
                            "Image2": float(img2_val),
                            "Detectability": detect_msg.replace("Image 2 ", "")
                        }

                writer.writerow([
                    cover_image, phase_runner[0],
                    psnr, ssim, mse, entropy_diff,
                    chi_data["Blue"].get("Image1"), chi_data["Blue"].get("Image2"), chi_data["Blue"].get("Detectability"),
                    chi_data["Green"].get("Image1"), chi_data["Green"].get("Image2"), chi_data["Green"].get("Detectability"),
                    chi_data["Red"].get("Image1"), chi_data["Red"].get("Image2"), chi_data["Red"].get("Detectability"),
                ])

if __name__ == "__main__":
    # Expect the cover image path as an argument
    if len(sys.argv) != 2:
        print("Usage: python master_runner.py <image_src_path>")
        main()
    else:
        main(sys.argv[1])