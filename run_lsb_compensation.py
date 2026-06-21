import cv2
import os
import sys
from src.stego.lsb_compensation import embed_message, extract_message

DEFAULT_MESSAGE = "The quick brown fox jumps over the lazy dog"


def run_lsb_compensation(cover_img_path, message=DEFAULT_MESSAGE, debug = False):
    output_dir = os.path.join("images", "output")

    img_name = os.path.splitext(os.path.basename(cover_img_path))[0]
    stego_img_path = os.path.join(output_dir, f"{img_name}_phase4_3.png")
    secret_message = message

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
    if len(sys.argv) < 2:
        print("Usage: python run_lsb_compensation.py <cover_image_path> [secret_message]")
        sys.exit(1)

    cover_img_path = sys.argv[1]
    message = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MESSAGE
    run_lsb_compensation(cover_img_path, message=message, debug=True)