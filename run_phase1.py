import os
import sys
from src.stego.lsb_basic import embed_message, extract_message

def run_phase1(cover_img_path, debug = False):
    output_dir = os.path.join("images", "output")

    img_name = os.path.splitext(os.path.basename(cover_img_path))[0]
    stego_img = os.path.join(output_dir, f"{img_name}_phase1.png")
    secret_message = "This is your boi Eswar!"

    if debug:
        print("[*] Embedding message...")
    metrics = embed_message(cover_img_path, secret_message, stego_img)
    if debug:
        print("[+] Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
    if debug:
        print("[*] Extracting message...")
    extracted = extract_message(stego_img, len(secret_message))
    if debug:
        print(f"[+] Extracted message: {extracted}")
    return metrics


if __name__ == "__main__":
    # Expect the cover image path as an argument
    if len(sys.argv) != 2:
        print("Usage: python runner_phase1.py <cover_image_path>")
        sys.exit(1)

    cover_img_path = sys.argv[1]
    run_phase1(cover_img_path, debug=True)