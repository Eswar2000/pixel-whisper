import cv2
import numpy as np
from PIL import Image
import json
import os
from src.utils.metrics import calculate_metrics

def logistic_map(seed, r, size):
    x = seed
    seq = []
    for _ in range(size):
        x = r * x * (1 - x)
        seq.append(x)
    return np.array(seq)

def get_edge_strength(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute Sobel gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Edge magnitude
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    # Normalize to [0,1]
    edge_norm = (edge_magnitude - edge_magnitude.min()) / (edge_magnitude.max() - edge_magnitude.min() + 1e-10)
    return edge_norm

def embed_message_content_aware(cover_path, message, output_image_path, meta_path, seed=0.54321, r=3.99, debug=False):
    # Load image
    image = cv2.imread(cover_path)
    if image is None:
        raise FileNotFoundError(f"Cover image not found: {cover_path}")

    message_bytes = message.encode("utf-8")
    message_length = len(message_bytes)
    flat_img = image.flatten()

    if message_length * 8 > len(flat_img):
        raise ValueError("Message too large for this image!")

    # Logistic sequence
    chaos_seq = logistic_map(seed, r, len(flat_img))

    # Edge strength for each pixel (grayscale, shape = number of pixels)
    edge_map = get_edge_strength(image).flatten()

    # Repeat edge_map per channel to match flattened image size
    edge_map = np.repeat(edge_map, image.shape[2])  # e.g., 3 for RGB

    # Normalize chaos to [0,1]
    chaos_norm = (chaos_seq - chaos_seq.min()) / (chaos_seq.max() - chaos_seq.min() + 1e-10)

    # Weighted score: weighted sum of edge + chaos
    alpha = 0.7  # weight for edge
    combined_score = alpha * edge_map + (1 - alpha) * chaos_norm

    # Sort pixel indices by descending combined score
    pixel_indices = np.argsort(-combined_score)

    # Prepare bits
    bits = np.unpackbits(np.frombuffer(message_bytes, dtype=np.uint8))

    # Embed bits into flattened image's LSB at pixel_indices
    for i, bit in enumerate(bits):
        idx = pixel_indices[i]
        flat_img[idx] = (flat_img[idx] & ~1) | bit

    # Reshape and save stego image
    stego_img = flat_img.reshape(image.shape)
    cv2.imwrite(output_image_path, stego_img)

    # Save meta data
    meta_data = {
        "seed": seed,
        "r": r,
        "message_length": message_length,
        "pixel_order": pixel_indices[:len(bits)].tolist(),
        "alpha": alpha,
    }
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)

    if debug:
        recovered = extract_message_content_aware(output_image_path, meta_path, debug=False)
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

def extract_message_content_aware(stego_path, meta_path, debug=False):
    image = cv2.imread(stego_path)
    if image is None:
        raise FileNotFoundError(f"Stego image not found: {stego_path}")

    flat_img = image.flatten()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    message_length = meta["message_length"]
    pixel_order = meta["pixel_order"]

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
        return recovered_bytes

    return recovered_message
