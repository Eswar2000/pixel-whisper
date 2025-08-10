import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from src.utils.metrics import calculate_metrics


def message_to_bits(message: str):
    """Convert string message to list of bits."""
    return [int(bit) for char in message for bit in f"{ord(char):08b}"]


def bits_to_message(bits):
    """Convert list of bits to string message."""
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join([chr(int(''.join(map(str, byte)), 2)) for byte in chars])


def embed_message(cover_image_path, message, stego_image_path):
    """Embed a text message into an image using LSB."""
    img = Image.open(cover_image_path).convert("RGB")
    arr = np.array(img)

    bits = message_to_bits(message)
    total_pixels = arr.size // 3  # total RGB pixels
    if len(bits) > total_pixels:
        raise ValueError("Message is too long for this image.")

    # Flatten and modify LSB
    flat = arr.flatten()
    for i, bit in enumerate(bits):
        flat[i] = (flat[i] & ~1) | bit  # clear last bit, set to message bit

    arr_stego = flat.reshape(arr.shape)
    stego_img = Image.fromarray(arr_stego.astype(np.uint8))
    stego_img.save(stego_image_path)

    # Calculate metrics
    original = np.array(Image.open(cover_image_path))
    stego = np.array(Image.open(stego_image_path))
    metrics = calculate_metrics(original, stego)
    return metrics


def extract_message(stego_image_path, message_length):
    """Extract hidden message from image."""
    img = Image.open(stego_image_path).convert("RGB")
    arr = np.array(img)
    flat = arr.flatten()

    bits = [flat[i] & 1 for i in range(message_length * 8)]
    return bits_to_message(bits)