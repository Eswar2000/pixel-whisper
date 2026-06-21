import os
import sys
import numpy as np
from src.stego.lsb_basic import embed_message, extract_message
from src.chaos.logistic_map import chaotic_encrypt, chaotic_decrypt

DEFAULT_MESSAGE = "This is your boi Eswar!"


def text_to_bits(text):
    return np.array([int(b) for char in text for b in format(ord(char), '08b')], dtype=np.uint8)

def bits_to_text(bits):
    chars = [chr(int(''.join(str(bit) for bit in bits[i:i+8]), 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars)

def run_chaotic_encrypt_lsb(cover_img_path, message=DEFAULT_MESSAGE, debug = False):
    output_dir = os.path.join("images", "output")

    img_name = os.path.splitext(os.path.basename(cover_img_path))[0]
    stego_img = os.path.join(output_dir, f"{img_name}_phase2.png")
    secret_message = message
    message_bits = text_to_bits(secret_message)

    # Chaotic encryption
    encrypted_bits = chaotic_encrypt(message_bits, x0=0.73456, r=3.6125)

    # Embed encrypted bits
    if debug:
        print("[*] Embedding message...")
    metrics = embed_message(cover_img_path, ''.join(str(b) for b in encrypted_bits), stego_img)
    if debug:
        print("[+] Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    # Extract bits back
    if debug:
        print("[*] Extracting message...")
    extracted_bits_str = extract_message(stego_img, len(encrypted_bits))
    extracted_bits = np.array([int(b) for b in extracted_bits_str], dtype=np.uint8)

    # Chaotic decryption
    decrypted_bits = chaotic_decrypt(extracted_bits, x0=0.73456, r=3.6125)
    recovered_message = bits_to_text(decrypted_bits)
    if debug:
        print(f"[+] Extracted message: {recovered_message}")
    return metrics


if __name__ == "__main__":
    # Expect the cover image path as an argument, with an optional secret message
    if len(sys.argv) < 2:
        print("Usage: python run_chaotic_encrypt_lsb.py <cover_image_path> [secret_message]")
        sys.exit(1)

    cover_img_path = sys.argv[1]
    message = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MESSAGE
    run_chaotic_encrypt_lsb(cover_img_path, message=message, debug=True)