import os
import sys
from src.stego.chaos_dynamic_lsb import embed_message_dynamic, extract_message_dynamic

DEFAULT_MESSAGE = "The quick brown fox jumps over the lazy dog"


def run_chaos_dynamic_lsb(cover_img_path, message=DEFAULT_MESSAGE, debug = False):
    output_dir = os.path.join("images", "output")

    img_name = os.path.splitext(os.path.basename(cover_img_path))[0]
    stego_img = os.path.join(output_dir, f"{img_name}_phase3.png")

    meta_file = "images/output/meta/meta_phase3.json"
    secret_message = message

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
    if len(sys.argv) < 2:
        print("Usage: python run_chaos_dynamic_lsb.py <cover_image_path> [secret_message]")
        sys.exit(1)

    cover_img_path = sys.argv[1]
    message = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MESSAGE
    run_chaos_dynamic_lsb(cover_img_path, message=message, debug=True)