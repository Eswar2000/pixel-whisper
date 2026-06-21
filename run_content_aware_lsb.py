import os
import sys
from src.stego.chaos_content_aware_lsb import embed_message_content_aware, extract_message_content_aware

DEFAULT_MESSAGE = "The quick brown fox jumps over the lazy dog"


def run_content_aware_lsb(cover_img_path, message=DEFAULT_MESSAGE, debug = False):
    output_dir = os.path.join("images", "output")

    img_name = os.path.splitext(os.path.basename(cover_img_path))[0]
    stego_img = os.path.join(output_dir, f"{img_name}_phase4_1.png")
    meta_file = "images/output/meta/meta_phase4_1.json"
    secret_message = message

    if debug:
        print("[*] Embedding message...")
    metrics = embed_message_content_aware(cover_img_path, secret_message, stego_img, meta_file, debug=debug)
    if debug:
        print("[+] Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    if debug:
        print("[*] Extracting message...")
    extracted = extract_message_content_aware(stego_img, meta_file, debug=debug)
    if debug:
        print(f"[+] Extracted message: {extracted}")
    return metrics

if __name__ == "__main__":
    # Expect the cover image path as an argument, with an optional secret message
    if len(sys.argv) < 2:
        print("Usage: python run_content_aware_lsb.py <cover_image_path> [secret_message]")
        sys.exit(1)

    cover_img_path = sys.argv[1]
    message = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MESSAGE
    run_content_aware_lsb(cover_img_path, message=message, debug=True)