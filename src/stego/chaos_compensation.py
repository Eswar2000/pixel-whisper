import numpy as np
from PIL import Image
import hashlib
from src.utils.metrics import calculate_metrics

def logistic_map(x, r=3.99):
    return r * x * (1 - x)

def seed_to_x0(seed: str):
    # Hash the seed to int, then scale to (0,1)
    h = hashlib.sha256(seed.encode('utf-8')).hexdigest()
    val = int(h[:8], 16)  # first 8 hex digits to int
    return (val % 9999 + 1) / 10000  # (0.0001 to 0.9999)

def get_chaotic_values(width, height, seed):
    x = seed_to_x0(seed)
    values = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            x = logistic_map(x)
            values[i, j] = x
    return values

def bit_get(value, bit_index):
    return (value >> bit_index) & 1

def bit_set(value, bit_index, bit):
    if bit == 1:
        return value | (1 << bit_index)
    else:
        return value & ~(1 << bit_index)

def embed_bit_in_pixel(pixel, bit, embed_channel, comp_channel):
    # pixel is (R,G,B)
    pixel = list(pixel)
    original = pixel[embed_channel]
    original_comp = pixel[comp_channel]

    # Extract LSB of embed channel
    lsb = original & 1

    if lsb != bit:
        # Flip LSB in embed channel
        pixel[embed_channel] = bit_set(original, 0, bit)
        # Compensate by flipping LSB in comp channel
        comp_lsb = original_comp & 1
        pixel[comp_channel] = bit_set(original_comp, 0, 1 - comp_lsb)
        flipped = True
    else:
        flipped = False

    return tuple(pixel), flipped

def extract_bit_from_pixel(pixel, embed_channel):
    return pixel[embed_channel] & 1

def embed_message(image_path, message, output_path, seed="defaultpassword"):
    img = Image.open(image_path)
    img = img.convert('RGB')
    pixels = np.array(img)
    height, width, _ = pixels.shape

    # Convert message to bits
    message_bytes = message.encode('utf-8')
    message_bits = []
    for byte in message_bytes:
        for i in range(8):
            message_bits.append((byte >> (7 - i)) & 1)

    max_bits = height * width
    if len(message_bits) > max_bits:
        raise ValueError("Message too long to embed in image")

    chaos_values = get_chaotic_values(width, height, seed)

    bit_idx = 0
    for i in range(height):
        for j in range(width):
            if bit_idx >= len(message_bits):
                break
            c = chaos_values[i, j]

            # Decide embed channel and compensation channel based on chaotic value
            if c < 0.33:
                embed_channel = 0  # R
                comp_channel = 1   # G
            elif c < 0.66:
                embed_channel = 1  # G
                comp_channel = 2   # B
            else:
                embed_channel = 2  # B
                comp_channel = 0   # R

            bit_to_embed = message_bits[bit_idx]

            pixel = tuple(pixels[i, j])
            new_pixel, flipped = embed_bit_in_pixel(pixel, bit_to_embed, embed_channel, comp_channel)
            pixels[i, j] = new_pixel
            bit_idx += 1
        if bit_idx >= len(message_bits):
            break

    # Save stego image
    stego_img = Image.fromarray(pixels)
    stego_img.save(output_path)

    # Calculate metrics
    original = np.array(Image.open(image_path))
    stego = np.array(Image.open(output_path))
    metrics = calculate_metrics(original, stego)
    return metrics

def extract_message(image_path, message_length, seed="defaultpassword"):
    img = Image.open(image_path)
    img = img.convert('RGB')
    pixels = np.array(img)
    height, width, _ = pixels.shape

    chaos_values = get_chaotic_values(width, height, seed)

    bits_needed = message_length * 8
    extracted_bits = []

    bit_idx = 0
    for i in range(height):
        for j in range(width):
            if bit_idx >= bits_needed:
                break
            c = chaos_values[i, j]

            if c < 0.33:
                embed_channel = 0  # R
            elif c < 0.66:
                embed_channel = 1  # G
            else:
                embed_channel = 2  # B

            pixel = tuple(pixels[i, j])
            bit = extract_bit_from_pixel(pixel, embed_channel)
            extracted_bits.append(bit)
            bit_idx += 1
        if bit_idx >= bits_needed:
            break

    # Convert bits back to string
    bytes_out = []
    for i in range(0, len(extracted_bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | extracted_bits[i + j]
        bytes_out.append(byte)
    try:
        message = bytes(bytes_out).decode('utf-8')
    except UnicodeDecodeError:
        message = "Error decoding message. Wrong seed or corrupted data."
    return message
