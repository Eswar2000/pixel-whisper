import cv2
import sys
import os
from src.stego.chaos_compensation import embed_message, extract_message

def run_phase4_2(cover_img_path, debug = False):
    output_dir = os.path.join("images", "output")

    img_name = os.path.splitext(os.path.basename(cover_img_path))[0]
    stego_img = os.path.join(output_dir, f"{img_name}_phase4_2.png")
    secret_message = "This is your boi Eswar!"
    seed = "sussy-chungus"

    # Embed
    if debug:
        print("[*] Embedding message...")
    metrics = embed_message(cover_img_path, secret_message, stego_img, seed)
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
    extracted = extract_message(stego_img, len(secret_message), seed)
    if debug:
        print("[+] Extracted message:", extracted)
    return metrics

if __name__ == "__main__":
    # Expect the cover image path as an argument
    if len(sys.argv) != 2:
        print("Usage: python runner_phase4_2.py <cover_image_path>")
        sys.exit(1)

    cover_img_path = sys.argv[1]
    run_phase4_2(cover_img_path, debug=True)