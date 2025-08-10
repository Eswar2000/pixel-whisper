import cv2
import numpy as np
from PIL import Image
from src.utils.metrics import calculate_metrics

# Embed a message into image with channel control and cross-channel compensation
def embed_message(cover_path, output_image_path, message, channel_to_use=2):  # 0=Blue,1=Green,2=Red
    image = cv2.imread(cover_path)
    # Convert message to bits
    bits = []
    for c in message:
        bits.extend([int(b) for b in format(ord(c), '08b')])

    img = image.copy()
    rows, cols, _ = img.shape
    max_bits = rows * cols
    if len(bits) > max_bits:
        raise ValueError("Message too long to embed in image")

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(bits):
                break
            pixel = img[r, c].copy()
            original_val = pixel[channel_to_use]
            bit = bits[idx]

            # Current LSB
            current_lsb = original_val & 1
            if current_lsb != bit:
                # Flip bit using compensation from other channels
                # We'll reduce by 1 if possible, else increase by 1
                # and try to keep the sum parity stable via compensation
                if original_val > 0:
                    pixel[channel_to_use] = original_val - 1
                else:
                    pixel[channel_to_use] = original_val + 1

                # Cross-channel compensation: flip LSB of another channel to balance parity
                for ch in range(3):
                    if ch != channel_to_use:
                        # Flip the LSB of this channel to compensate parity if needed
                        other_val = pixel[ch]
                        pixel[ch] = other_val ^ 1  # flip LSB
                        break

            img[r, c] = pixel
            idx += 1
        if idx >= len(bits):
            break
    cv2.imwrite(output_image_path, img)

    # Calculate metrics
    original = np.array(Image.open(cover_path))
    stego = np.array(Image.open(output_image_path))
    metrics = calculate_metrics(original, stego)
    return metrics

# Extract message from image using known channel
def extract_message(stego_path, message_length, channel_to_use=2):
    image = cv2.imread(stego_path)
    bits = []
    rows, cols, _ = image.shape
    total_bits = message_length * 8
    idx = 0

    for r in range(rows):
        for c in range(cols):
            if idx >= total_bits:
                break
            pixel = image[r, c]
            bit = pixel[channel_to_use] & 1
            bits.append(bit)
            idx += 1
        if idx >= total_bits:
            break

    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        byte_val = 0
        for bit in byte:
            byte_val = (byte_val << 1) | bit
        chars.append(chr(byte_val))
    return ''.join(chars)
