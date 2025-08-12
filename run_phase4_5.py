# run_phase_acme.py
import os
import sys
from src.stego.acme import embed_message, extract_message

def run_phase_4_5(cover_img_path, debug=False):
    output_dir = os.path.join("images", "output")
    meta_dir = os.path.join("images", "output", "meta")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    img_name = os.path.splitext(os.path.basename(cover_img_path))[0]
    stego_img_path = os.path.join(output_dir, f"{img_name}_phase4_5.png")
    meta_file = os.path.join(meta_dir, f"meta_phase4_5.json")
    secret_message = "This is your boi Eswar!"

    if debug:
        print("[*] Embedding message with ACME...")
    metrics = embed_message(cover_img_path, secret_message, stego_img_path, meta_file,
                            method="acme", seed_str="sussy-chungus", logistic_seed=0.54321,
                            logistic_r=3.99, alpha=0.7, max_bits_per_pixel=2, debug=debug)
    if debug:
        print("[+] Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")

    if debug:
        print("[*] Extracting message...")
    extracted = extract_message(stego_img_path, meta_file)
    if debug:
        print("[+] Extracted message:", extracted)
    return metrics

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_phase_acme.py <cover_image_path>")
        sys.exit(1)
    cover_img_path = sys.argv[1]
    run_phase_4_5(cover_img_path, debug=True)
