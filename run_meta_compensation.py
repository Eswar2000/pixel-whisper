import os
import sys
from src.stego.meta_compensation import embed_message, extract_message

DEFAULT_MESSAGE = "The quick brown fox jumps over the lazy dog"


def run_meta_compensation(cover_img_path, message=DEFAULT_MESSAGE, debug = False):
    output_dir = os.path.join("images", "output")

    img_name = os.path.splitext(os.path.basename(cover_img_path))[0]
    stego_img_path = os.path.join(output_dir, f"{img_name}_phase4_4.png")
    meta_file = "images/output/meta/meta_phase4_4.json"
    secret_message = message
    

    # Embed with context-aware method
    if debug:
        print("[*] Embedding message...")
    metrics = embed_message(cover_img_path, secret_message, stego_img_path, meta_file, method="super_model", seed_str="sussy-chungus", logistic_seed=0.54321, logistic_r=3.99, alpha=0.7)
    if debug:
        print("[+] Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    # Extract message
    if debug:
        print("[*] Extracting message...")
    extracted = extract_message(stego_img_path, meta_file)
    if debug:
        print("[+] Extracted message:", extracted)
    return metrics

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_meta_compensation.py <cover_image_path> [secret_message]")
        sys.exit(1)

    cover_img_path = sys.argv[1]
    message = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MESSAGE
    run_meta_compensation(cover_img_path, message=message, debug=True)