import cv2
from io import BytesIO
from PIL import Image
import numpy as np

def jpeg_compression(img: np.ndarray, quality: int = 85) -> np.ndarray:
    """Apply JPEG recompression at given quality."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', img, encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec

def add_gaussian_noise(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Add Gaussian noise (σ on 0–255 scale)."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)

def gaussian_blur(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur with kernel size and sigma."""
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply median filtering."""
    return cv2.medianBlur(img, ksize)

def bilateral_filter(img: np.ndarray, d: int = 5, sigmaColor: float = 75, sigmaSpace: float = 75) -> np.ndarray:
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

def non_local_means_filter(img: np.ndarray, h: float = 10.0) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

def gamma_correction(img: np.ndarray, gamma: float = 0.9) -> np.ndarray:
    """Apply gamma correction."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i/255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

import numpy as np

def string_ber(original: str, recovered: str, encoding="utf-8") -> float:
    """
    Compute Bit Error Rate (BER) between two strings.
    Strings are first converted to bytes using the given encoding,
    then to bits for comparison.
    """
    # Convert strings to bytes
    orig_bytes = original.encode(encoding, errors="replace")
    try:
        rec_bytes = recovered.encode(encoding, errors="replace")
    except Exception:
        rec_bytes = recovered

    # Convert bytes to bits
    orig_bits = np.unpackbits(np.frombuffer(orig_bytes, dtype=np.uint8))
    rec_bits = np.unpackbits(np.frombuffer(rec_bytes, dtype=np.uint8))

    # Pad the shorter one with zeros (strict comparison)
    max_len = max(len(orig_bits), len(rec_bits))
    if len(orig_bits) < max_len:
        orig_bits = np.pad(orig_bits, (0, max_len - len(orig_bits)))
    if len(rec_bits) < max_len:
        rec_bits = np.pad(rec_bits, (0, max_len - len(rec_bits)))

    # Compute BER
    min_len = min(len(orig_bits), len(rec_bits))
    errors = np.count_nonzero(orig_bits[:min_len] != rec_bits[:min_len])
    # errors = np.count_nonzero(orig_bits != rec_bits)
    ber = errors / max_len
    return ber
