import cv2
import numpy as np
from PIL import Image
import json
import os
from src.utils.metrics import calculate_metrics


def logistic_map(seed, r, size):
    """
    Generate a chaotic sequence using the logistic map.
    seed: float between 0 and 1
    r: control parameter (usually 3.5 < r < 4.0 for chaos)
    size: length of sequence
    """
    x = seed
    seq = []
    for _ in range(size):
        x = r * x * (1 - x)
        seq.append(x)
    return np.array(seq)


def embed_message_dynamic(cover_path, message, output_image_path, meta_path, seed=0.54321, r=3.99, debug=False):
    """
    Embed a message using dynamic pixel selection with a chaotic map.
    Stores meta info in JSON for extraction.
    """
    # Read cover image
    image = cv2.imread(cover_path)
    if image is None:
        raise FileNotFoundError(f"Cover image not found: {cover_path}")

    # Convert message to bytes
    message_bytes = message.encode("utf-8")
    message_length = len(message_bytes)

    # Flatten image to 1D for LSB modification
    flat_img = image.flatten()

    if message_length * 8 > len(flat_img):
        raise ValueError("Message too large for this image!")

    # Generate chaotic sequence and get pixel index order
    seq = logistic_map(seed, r, len(flat_img))
    pixel_indices = np.argsort(seq)

    # Embed message bits
    bits = np.unpackbits(np.frombuffer(message_bytes, dtype=np.uint8))
    for i, bit in enumerate(bits):
        flat_img[pixel_indices[i]] = (flat_img[pixel_indices[i]] & ~1) | bit

    # Save stego image
    stego_img = flat_img.reshape(image.shape)
    cv2.imwrite(output_image_path, stego_img)

    # Save metadata
    meta_data = {
        "seed": seed,
        "r": r,
        "message_length": message_length,
        "pixel_order": pixel_indices[:len(bits)].tolist()
    }
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)

    # Optional debug check — try extracting immediately
    if debug:
        recovered = extract_message_dynamic(output_image_path, meta_path, debug=False)
        if recovered != message:
            print("[!] DEBUG WARNING: Extracted message does NOT match original!")
            print(f"    Extracted: {recovered}")
            print(f"    Original:  {message}")
        else:
            print("[+] DEBUG: Extraction OK, matches original.")
    
    # Calculate metrics
    original = np.array(Image.open(cover_path))
    stego = np.array(Image.open(output_image_path))
    metrics = calculate_metrics(original, stego)
    return metrics


def extract_message_dynamic(stego_path, meta_path, debug=False):
    """
    Extract a hidden message using metadata.
    """
    # Load stego image
    image = cv2.imread(stego_path)
    if image is None:
        raise FileNotFoundError(f"Stego image not found: {stego_path}")

    flat_img = image.flatten()

    # Load metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    message_length = meta["message_length"]
    pixel_order = meta["pixel_order"]

    # Extract bits
    bits = []
    for idx in pixel_order:
        bits.append(flat_img[idx] & 1)

    bits = np.array(bits[:message_length * 8], dtype=np.uint8)
    recovered_bytes = np.packbits(bits).tobytes()

    try:
        recovered_message = recovered_bytes.decode("utf-8")
    except UnicodeDecodeError:
        if debug:
            print("[!] Could not decode as UTF-8, returning raw bytes")
        return recovered_bytes  # return raw bytes if decoding fails

    return recovered_message
