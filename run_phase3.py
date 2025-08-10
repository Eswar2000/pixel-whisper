import os
import sys
from src.stego.chaos_dynamic_lsb import embed_message_dynamic, extract_message_dynamic

def run_phase3(cover_img_path, debug = False):
    output_dir = os.path.join("images", "output")

    img_name = os.path.splitext(os.path.basename(cover_img_path))[0]
    stego_img = os.path.join(output_dir, f"{img_name}_phase3.png")

    meta_file = "images/output/meta/meta_phase3.json"
    secret_message = "This is your boi Eswar!"

    if debug:
        print("[*] Embedding message...")
    metrics = embed_message_dynamic(cover_img_path, secret_message, stego_img, meta_file, debug=debug)
    if debug:
        print("[+] Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    if debug:
        print("[*] Extracting message...")
    extracted = extract_message_dynamic(stego_img, meta_file, debug=debug)
    if debug:
        print(f"[+] Extracted message: {extracted}")
    return metrics

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python runner_phase3.py <cover_image_path>")
        sys.exit(1)

    cover_img_path = sys.argv[1]
    run_phase3(cover_img_path, debug=True)