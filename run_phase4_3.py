import cv2
import os
import sys
from src.stego.lsb_compensation import embed_message, extract_message

def run_phase4_3(cover_img_path, debug = False):
    output_dir = os.path.join("images", "output")

    img_name = os.path.splitext(os.path.basename(cover_img_path))[0]
    stego_img_path = os.path.join(output_dir, f"{img_name}_phase4_3.png")
    secret_message = "This is your boi Eswar!"

    # Embed
    if debug:
        print("[*] Embedding message...")
    metrics = embed_message(cover_img_path, stego_img_path, secret_message, channel_to_use=0)
    if debug:
        print("[+] Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    # Extract
    if debug:
        print("[*] Extracting message...")
    extracted_message = extract_message(stego_img_path, len(secret_message), channel_to_use=0)
    if debug:
        print(f"[+] Extracted message: {extracted_message}")
    return metrics

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python runner_phase4_3.py <cover_image_path>")
        sys.exit(1)

    cover_img_path = sys.argv[1]
    run_phase4_3(cover_img_path, debug=True)