# src/stego/acme.py
import os
import json
import hashlib
import numpy as np
import cv2
from PIL import Image
from src.utils.metrics import calculate_metrics

# -----------------------
# Tunable parameters
# -----------------------
DEFAULT_MAX_BITS_PER_PIXEL = 2   # allow 0/1/2 bits depending on perceptual mask
DEFAULT_ALPHA = 0.7             # weight for edge vs chaos in ordering
DEFAULT_LOGISTIC_R = 3.99
DEFAULT_LOGISTIC_SEED = 0.54321
DEFAULT_BLOCK_SEARCH_RADIUS = 1  # neighborhood radius to search for compensator
# -----------------------


def _logistic_map_iter(x, r=DEFAULT_LOGISTIC_R, steps=1):
    for _ in range(steps):
        x = r * x * (1 - x)
    return x


def _seed_to_x0(seed_str):
    h = hashlib.sha256(seed_str.encode('utf-8')).hexdigest()
    val = int(h[:8], 16)
    return ((val % 9999) + 1) / 10000.0


def _gen_chaos_streams(seed_str, length, r=DEFAULT_LOGISTIC_R, n_streams=3):
    """
    Generate 'n_streams' chaotic sequences of given length using logistic map.
    Returns ndarray shape (n_streams, length) with values in (0,1).
    """
    x0 = _seed_to_x0(seed_str)
    streams = np.zeros((n_streams, length), dtype=np.float64)
    x = x0
    for s in range(n_streams):
        for i in range(length):
            x = _logistic_map_iter(x, r)
            streams[s, i] = x
        # perturb x a bit for next stream
        x = (x + 0.123456789) % 1.0
    # normalize each stream
    streams = (streams - streams.min(axis=1, keepdims=True)) / (
        streams.max(axis=1, keepdims=True) - streams.min(axis=1, keepdims=True) + 1e-12
    )
    return streams


def _edge_texture_mask(image_bgr):
    """
    Compute perceptual mask combining Sobel edge and local variance (texture).
    Returns mask in [0,1] per-pixel.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Sobel magnitude
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sx**2 + sy**2)
    sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min() + 1e-12)

    # Local variance (texture): use 7x7 window
    k = 7
    mean = cv2.blur(gray, (k, k))
    mean_sq = cv2.blur(gray * gray, (k, k))
    var = np.maximum(mean_sq - mean * mean, 0.0)
    var_n = (var - var.min()) / (var.max() - var.min() + 1e-12)

    mask = 0.6 * sobel + 0.4 * var_n
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-12)
    return mask


def _bits_from_message(message: str):
    b = np.frombuffer(message.encode('utf-8'), dtype=np.uint8)
    bits = np.unpackbits(b)
    return bits.astype(np.uint8)


def _message_from_bits(bits):
    # bits is array-like length multiple of 8
    bits = np.array(bits, dtype=np.uint8)
    r = len(bits) % 8
    if r != 0:
        bits = np.concatenate([bits, np.zeros(8 - r, dtype=np.uint8)])
    b = np.packbits(bits)
    try:
        return b.tobytes().decode('utf-8')
    except UnicodeDecodeError:
        return b.tobytes()


def _set_lsb(val, bit):
    val = int(val)
    if bit == 1:
        return val | 1
    else:
        return val & ~1


def _find_compensator(pixels, y, x, forbidden_channels, max_radius=DEFAULT_BLOCK_SEARCH_RADIUS):
    """
    Search small neighborhood for a compensator pixel and channel to flip to balance parity.
    Returns (yy, xx, comp_ch) or None
    forbidden_channels: set() of channels to avoid (e.g., channels that will be used later in same pixel)
    """
    h, w, _ = pixels.shape
    for r in range(1, max_radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                yy = y + dy
                xx = x + dx
                if yy < 0 or xx < 0 or yy >= h or xx >= w:
                    continue
                if yy == y and xx == x:
                    continue
                for ch in range(3):
                    if ch in forbidden_channels:
                        continue
                    # safe check - always valid
                    return yy, xx, ch
    return None


# -------------------------
# Embedding (main function)
# -------------------------
def embed_acme(
    cover_path,
    message,
    out_path,
    meta_path,
    seed_str="acme_seed",
    logistic_seed=DEFAULT_LOGISTIC_SEED,
    logistic_r=DEFAULT_LOGISTIC_R,
    alpha=DEFAULT_ALPHA,
    max_bits_per_pixel=DEFAULT_MAX_BITS_PER_PIXEL,
    debug=False,
):
    """
    Embed message into cover image using adaptive chaotic multi-map embedding.
    Returns metrics dict (via your project's calculate_metrics).
    """

    # Read image (use cv2 for mask, PIL for pixel manipulation)
    cover_bgr = cv2.imread(cover_path)
    if cover_bgr is None:
        raise FileNotFoundError(f"Cover image not found: {cover_path}")
    h, w, ch = cover_bgr.shape
    assert ch == 3

    bits = _bits_from_message(message)
    n_bits = len(bits)

    mask = _edge_texture_mask(cover_bgr)  # shape (h,w), [0,1]

    n_pixels = h * w
    streams = _gen_chaos_streams(seed_str + "_acme", n_pixels, r=logistic_r, n_streams=3)

    stream0 = streams[0, :].reshape((h, w))
    priority = alpha * mask + (1.0 - alpha) * stream0
    flat_priority = priority.flatten()
    pixel_ord = np.argsort(-flat_priority)

    stream1 = streams[1, :].reshape((h, w))
    cap_map = np.zeros((h, w), dtype=np.int32)
    for y in range(h):
        for x in range(w):
            pval = mask[y, x]
            s1 = stream1[y, x]
            cap = 1
            if max_bits_per_pixel >= 2 and (pval > 0.6 or s1 > 0.75):
                cap = 2
            cap_map[y, x] = cap

    stream2 = streams[2, :].reshape((h, w))

    pil = Image.open(cover_path).convert("RGB")
    pixels = np.array(pil)  # shape (h,w,3), dtype=uint8

    bit_idx = 0
    embedded_positions = []  # list of [y,x, [ch0,ch1,...]] in order

    for flat_idx in pixel_ord:
        if bit_idx >= n_bits:
            break
        y = flat_idx // w
        x = flat_idx % w
        cap = int(cap_map[y, x])
        sval = stream2[y, x]
        if sval < 1 / 3:
            embed_ch = 0
        elif sval < 2 / 3:
            embed_ch = 1
        else:
            embed_ch = 2

        channels_used = []
        # prepare list of future target channels to avoid when selecting compensator
        future_targets_all = [(embed_ch + s) % 3 for s in range(cap)]

        for slot in range(cap):
            if bit_idx >= n_bits:
                break

            target_ch = (embed_ch + slot) % 3
            target_bit = int(bits[bit_idx])

            orig_val = int(pixels[y, x, target_ch])
            orig_lsb = orig_val & 1
            if orig_lsb == target_bit:
                channels_used.append(int(target_ch))
                bit_idx += 1
                continue

            # set LSB in chosen target channel
            pixels[y, x, target_ch] = _set_lsb(orig_val, target_bit)

            # Compensator selection: avoid target_ch and channels that will be used later in this pixel.
            forbidden = set(channels_used)
            # Also forbid channels that are in future targets excluding current slot (they may be used later)
            future_targets = set(future_targets_all[slot + 1 :]) if slot + 1 < len(future_targets_all) else set()
            forbidden = forbidden.union(future_targets)
            forbidden.add(target_ch)

            comp_done = False
            # Prefer compensation inside same pixel in a channel that is not forbidden
            for comp_ch in range(3):
                if comp_ch in forbidden:
                    continue
                comp_orig = int(pixels[y, x, comp_ch])
                pixels[y, x, comp_ch] = _set_lsb(comp_orig, 1 - (comp_orig & 1))
                comp_done = True
                break

            if not comp_done:
                # fallback search neighbour
                comp = _find_compensator(pixels, y, x, forbidden, max_radius=DEFAULT_BLOCK_SEARCH_RADIUS)
                if comp is not None:
                    yy, xx, comp_ch = comp
                    comp_orig = int(pixels[yy, xx, comp_ch])
                    pixels[yy, xx, comp_ch] = _set_lsb(comp_orig, 1 - (comp_orig & 1))
                    comp_done = True
                # if still not found, we proceed without compensation (rare)

            channels_used.append(int(target_ch))
            bit_idx += 1

        if len(channels_used) > 0:
            embedded_positions.append([int(y), int(x), [int(ch) for ch in channels_used]])

    if bit_idx < n_bits:
        raise ValueError("Image too small to embed full message with current ACME parameters")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stego_img = Image.fromarray(pixels.astype(np.uint8))
    stego_img.save(out_path)

    meta = {
        "method": "acme",
        "seed_str": seed_str,
        "logistic_seed": logistic_seed,
        "logistic_r": logistic_r,
        "alpha": float(alpha),
        "max_bits_per_pixel": int(max_bits_per_pixel),
        "image_shape": [int(h), int(w), 3],
        "embedded_positions": embedded_positions,
        "n_message_bits": int(n_bits),
    }
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    original = np.array(Image.open(cover_path))
    stego = np.array(Image.open(out_path))
    metrics = calculate_metrics(original, stego)
    if debug:
        print(f"[acme] embedded bits: {n_bits}, used pixels: {len(embedded_positions)}")
    return metrics


# -------------------------
# Extraction
# -------------------------
def extract_acme(stego_path, meta_path, debug=False):
    """
    Extract message string from stego using metadata.
    """
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("method") != "acme":
        raise ValueError("Metadata method mismatch (expected 'acme')")

    h, w, channels = meta["image_shape"]
    embedded_positions = meta["embedded_positions"]
    n_bits = int(meta["n_message_bits"])

    img = Image.open(stego_path).convert("RGB")
    pixels = np.array(img)

    bits = []
    for item in embedded_positions:
        # item is [y, x, [ch0, ch1, ...]]
        if len(item) != 3:
            # backward-compat fallback: older format
            try:
                y, x, used_bits, embed_ch = item
                for _ in range(int(used_bits)):
                    bit = int(pixels[y, x, embed_ch]) & 1
                    bits.append(bit)
                    if len(bits) >= n_bits:
                        break
            except Exception:
                continue
        else:
            y, x, channels_list = item
            for ch in channels_list:
                bit = int(pixels[int(y), int(x), int(ch)]) & 1
                bits.append(int(bit))
                if len(bits) >= n_bits:
                    break
        if len(bits) >= n_bits:
            break

    bits = bits[:n_bits]
    message = _message_from_bits(bits)
    if debug:
        print("[acme] Extracted bits:", len(bits))
    return message


# -------------------------
# Wrapper to match project pattern
# -------------------------
def embed_message(cover_path, message, out_path, meta_path, method="acme", **kwargs):
    if method != "acme":
        raise NotImplementedError("acme wrapper: only method='acme' supported")
    return embed_acme(cover_path, message, out_path, meta_path, **kwargs)


def extract_message(stego_path, meta_path):
    return extract_acme(stego_path, meta_path)
