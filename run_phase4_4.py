import os
import sys
from src.stego.meta_compensation import embed_message, extract_message

def run_phase4_4(cover_img_path, debug = False):
    output_dir = os.path.join("images", "output")

    img_name = os.path.splitext(os.path.basename(cover_img_path))[0]
    stego_img_path = os.path.join(output_dir, f"{img_name}_phase4_4.png")
    meta_file = "images/output/meta/meta_phase4_4.json"
    secret_message = "This is your boi Eswar!"
    

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
    if len(sys.argv) != 2:
        print("Usage: python runner_phase4_4.py <cover_image_path>")
        sys.exit(1)

    cover_img_path = sys.argv[1]
    run_phase4_4(cover_img_path, debug=True)