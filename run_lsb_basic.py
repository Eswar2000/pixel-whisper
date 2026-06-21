import os
import sys
from src.stego.lsb_basic import embed_message, extract_message

DEFAULT_MESSAGE = "This is your boi Eswar!"


def run_lsb_basic(cover_img_path, message=DEFAULT_MESSAGE, debug = False):
    output_dir = os.path.join("images", "output")

    img_name = os.path.splitext(os.path.basename(cover_img_path))[0]
    stego_img = os.path.join(output_dir, f"{img_name}_phase1.png")
    secret_message = message

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
    # Expect the cover image path as an argument, with an optional secret message
    if len(sys.argv) < 2:
        print("Usage: python run_lsb_basic.py <cover_image_path> [secret_message]")
        sys.exit(1)

    cover_img_path = sys.argv[1]
    message = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MESSAGE
    run_lsb_basic(cover_img_path, message=message, debug=True)