import cv2
import numpy as np
import json
import os
from PIL import Image
import hashlib
from src.utils.metrics import calculate_metrics

# --- Helpers ---

def logistic_map_val(x, r=3.99):
    return r * x * (1 - x)

def logistic_map_seq(seed, r, size):
    x = seed
    seq = []
    for _ in range(size):
        x = logistic_map_val(x, r)
        seq.append(x)
    return np.array(seq)

def seed_to_x0(seed: str):
    h = hashlib.sha256(seed.encode('utf-8')).hexdigest()
    val = int(h[:8], 16)
    return (val % 9999 + 1) / 10000  # (0.0001, 0.9999)

def get_edge_strength(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_norm = (edge_magnitude - edge_magnitude.min()) / (edge_magnitude.max() - edge_magnitude.min() + 1e-10)
    return edge_norm

def bit_set(value, bit_index, bit):
    if bit == 1:
        return value | (1 << bit_index)
    else:
        return value & ~(1 << bit_index)

def embed_bit_in_pixel(pixel, bit, embed_channel, comp_channel):
    pixel = list(pixel)
    original = pixel[embed_channel]
    original_comp = pixel[comp_channel]

    lsb = original & 1

    if lsb != bit:
        pixel[embed_channel] = bit_set(original, 0, bit)
        comp_lsb = original_comp & 1
        pixel[comp_channel] = bit_set(original_comp, 0, 1 - comp_lsb)  # flip comp channel bit
        flipped = True
    else:
        flipped = False

    return tuple(pixel), flipped

def extract_bit_from_pixel(pixel, embed_channel):
    return pixel[embed_channel] & 1

def get_chaotic_values(width, height, seed_str, r=3.99):
    """
    Generate a 2D array (height x width) of chaotic logistic map values
    using a seed string to initialize the logistic map starting value.
    """
    x = seed_to_x0(seed_str)
    values = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            x = logistic_map_val(x, r)
            values[i, j] = x
    return values

# --- Main embedding method: Super Model ---

def embed_super_model(cover_path, message, output_image_path, meta_path,
                      seed_str="defaultpassword", logistic_seed=0.54321, logistic_r=3.99, alpha=0.7, debug=False):
    """
    Embeds message using:
    - Context-aware pixel ordering (edge + logistic chaos)
    - Cross-channel compensation (channel chosen per chaotic value)
    """

    # Read cover image using OpenCV for edge detection and pixel flattening
    image = cv2.imread(cover_path)
    if image is None:
        raise FileNotFoundError(f"Cover image not found: {cover_path}")
    height, width, channels = image.shape

    # Prepare message bits
    message_bytes = message.encode('utf-8')
    message_length = len(message_bytes)
    total_pixels = height * width * channels
    if message_length * 8 > height * width:
        raise ValueError("Message too large for image capacity (one bit per pixel)")

    # Step 1: Compute logistic map sequence for flattened pixels
    flat_len = height * width * channels
    chaos_seq = logistic_map_seq(logistic_seed, logistic_r, flat_len)

    # Step 2: Compute edge strength per pixel (grayscale)
    edge_map = get_edge_strength(image).flatten()
    # Repeat edge per channel to match logistic map length
    edge_map_full = np.repeat(edge_map, channels)

    # Step 3: Combine edge + chaos to get pixel importance scores
    chaos_norm = (chaos_seq - chaos_seq.min()) / (chaos_seq.max() - chaos_seq.min() + 1e-10)
    combined_score = alpha * edge_map_full + (1 - alpha) * chaos_norm

    # Step 4: Sort indices by descending importance (to pick pixels to embed in)
    sorted_indices = np.argsort(-combined_score)

    # Step 5: We'll embed one bit per pixel (not per channel), so for each pixel we select the RGB triple and
    # decide which channel to embed bit into + compensation channel based on chaotic values derived from logistic map on 2D pixel grid

    # Prepare PIL image for pixel-level channel manipulation
    pil_img = Image.open(cover_path).convert('RGB')
    pixels = np.array(pil_img)
    
    # Generate chaotic values for pixel positions (not channels)
    chaos_2d = get_chaotic_values(width, height, seed_str)

    # Convert message to bit array
    bits = np.unpackbits(np.frombuffer(message_bytes, dtype=np.uint8))
    bit_idx = 0

    # Keep track of which pixels we embedded in (for metadata)
    embedded_pixel_positions = []

    for flat_idx in sorted_indices:
        if bit_idx >= len(bits):
            break
        # Convert flat_idx to pixel coordinates and channel
        pixel_idx = flat_idx // channels
        channel_idx = flat_idx % channels
        y = pixel_idx // width
        x = pixel_idx % width

        # Only embed one bit per pixel, so embed when channel_idx==0 (red channel start) to avoid multiple bits per pixel
        # Alternatively, pick only one channel per pixel based on chaotic value (see below)
        if channel_idx != 0:
            continue

        # Use chaotic 2D value at (y,x) to decide embed_channel and comp_channel (like cross channel compensation)
        c = chaos_2d[y, x]
        if c < 0.33:
            embed_channel = 0  # R
            comp_channel = 1   # G
        elif c < 0.66:
            embed_channel = 1  # G
            comp_channel = 2   # B
        else:
            embed_channel = 2  # B
            comp_channel = 0   # R

        # Embed the bit at pixel[y,x] in embed_channel with compensation channel
        pixel = tuple(pixels[y, x])
        bit_to_embed = bits[bit_idx]
        new_pixel, flipped = embed_bit_in_pixel(pixel, bit_to_embed, embed_channel, comp_channel)
        pixels[y, x] = new_pixel
        embedded_pixel_positions.append([y, x])
        bit_idx += 1

    if bit_idx < len(bits):
        raise ValueError("Image too small to embed all bits with super model")

    # Save stego image
    stego_img = Image.fromarray(pixels)
    stego_img.save(output_image_path)

    # Save metadata for extraction
    meta_data = {
        "method": "super_model",
        "seed_str": seed_str,
        "logistic_seed": logistic_seed,
        "logistic_r": logistic_r,
        "alpha": alpha,
        "message_length": message_length,
        "embedded_pixels": embedded_pixel_positions,
        "image_shape": [height, width, channels]
    }

    # Convert embedded_pixels to a list of lists of int
    meta_data['embedded_pixels'] = [
        [int(coord) for coord in pair]
        for pair in meta_data['embedded_pixels']
    ]

    # Similarly, convert image_shape to native ints (if needed)
    meta_data['image_shape'] = [int(dim) for dim in meta_data['image_shape']]

    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=4)

    if debug:
        recovered = extract_super_model(output_image_path, meta_path)
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

# --- Extraction for Super Model ---

def extract_super_model(stego_path, meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if meta.get("method") != "super_model":
        raise ValueError("Meta data method mismatch: expected 'super_model'")

    seed_str = meta["seed_str"]
    logistic_seed = meta["logistic_seed"]
    logistic_r = meta["logistic_r"]
    alpha = meta["alpha"]
    message_length = meta["message_length"]
    embedded_pixel_positions = meta["embedded_pixels"]
    height, width, channels = meta["image_shape"]

    pil_img = Image.open(stego_path).convert('RGB')
    pixels = np.array(pil_img)

    chaos_2d = get_chaotic_values(width, height, seed_str)

    bits = []
    for y, x in embedded_pixel_positions:
        c = chaos_2d[y, x]
        if c < 0.33:
            embed_channel = 0
        elif c < 0.66:
            embed_channel = 1
        else:
            embed_channel = 2

        pixel = tuple(pixels[y, x])
        bit = extract_bit_from_pixel(pixel, embed_channel)
        bits.append(bit)

    bits = bits[:message_length * 8]
    bits = np.array(bits, dtype=np.uint8)
    recovered_bytes = np.packbits(bits).tobytes()
    try:
        recovered_message = recovered_bytes.decode('utf-8')
    except UnicodeDecodeError:
        recovered_message = "Error decoding message. Possibly wrong seed or corrupted."

    return recovered_message

# --- Unified interface ---

def embed_message(cover_path, message, output_path, meta_path, method="super_model", **kwargs):
    if method == "super_model":
        return embed_super_model(cover_path, message, output_path, meta_path, **kwargs)
    else:
        raise NotImplementedError(f"Embedding method '{method}' not implemented in super model file.")

def extract_message(stego_path, meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    method = meta.get("method", None)
    if method == "super_model":
        return extract_super_model(stego_path, meta_path)
    else:
        raise NotImplementedError(f"Extraction method '{method}' not implemented in super model file.")