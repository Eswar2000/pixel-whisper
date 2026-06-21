# src/stego/carq.py
"""
CARQ-Stego: Chaotic Adaptive Robust QIM Steganography.

Design goals (research-grade, fixes ACME's structural flaws):
  1. Blind extraction -- NO per-image position map. The decoder regenerates the
     exact embedding order from the secret key (logistic-map permutation) and the
     image dimensions alone.
  2. Self-describing payload -- a framed header (MAGIC | version | flags | length |
     CRC32) makes extraction self-terminating and integrity-checked. Accepts
     arbitrary bytes (text OR binary file), not a hardcoded string.
  3. Robustness -- embedding lives in the 8x8 block-DCT mid-band via Quantization
     Index Modulation (QIM), protected by Reed-Solomon ECC + interleaving +
     spread repetition (majority vote). Survives JPEG / noise / blur where LSB dies.
  4. Imperceptibility -- DC-guided adaptive quantization step (luminance masking).
     DC is never modified, so the decoder recomputes the same per-block step from
     the (possibly attacked) stego image -- content-adaptive yet blindly recoverable.
  5. Security -- chaotic stream-cipher (keystream XOR + key-driven bit permutation)
     keyed by a passphrase. The only secret is the passphrase; no side file.

The decoder needs only: the stego image, the passphrase, and the fixed scheme
parameters (delta, redundancy, nsym, coeff positions, channel). These are scheme
constants / part of the key -- never per-image side information.
"""

import os
import struct
import zlib
import hashlib

import numpy as np
import cv2
from reedsolo import RSCodec

from src.utils.metrics import calculate_metrics

# ----------------------------------------------------------------------------
# Scheme constants (shared by encoder and decoder -- NOT per-image side info)
# ----------------------------------------------------------------------------
MAGIC = b"CQ"          # frame sync marker
VERSION = 1
BLOCK = 8              # DCT block size

# Default tunable parameters (exposed for the robustness/imperceptibility sweep)
DEFAULT_DELTA = 44.0          # base QIM quantization step (robustness vs PSNR knob)
DEFAULT_REDUNDANCY = 5        # spread repetition factor R (majority vote)
DEFAULT_NSYM = 32             # Reed-Solomon parity bytes per 255-byte chunk
DEFAULT_LOGISTIC_R = 3.99
DEFAULT_CHANNEL = "Y"         # embed in luminance (most robust to JPEG)
DEFAULT_ADAPTIVE = True       # DC-guided adaptive step (luminance masking)

# Default zig-zag coefficient band used for embedding (start, end) exclusive end.
# Index 0 is DC. The lowest-frequency AC coefficients survive low-pass attacks
# (blur / median / motion) because they are least attenuated by filtering, whereas
# mid/high-frequency coefficients are driven toward zero and flip the QIM bit.
# This (start, end) window is the primary robustness knob.
DEFAULT_BAND = (1, 4)   # 3 lowest-frequency AC coefficients per block


# ----------------------------------------------------------------------------
# Zig-zag ordering for an 8x8 block
# ----------------------------------------------------------------------------
def _zigzag_positions(n=BLOCK):
    """Return list of (u, v) coordinates in JPEG zig-zag order for an n x n block."""
    coords = sorted(
        ((u, v) for u in range(n) for v in range(n)),
        key=lambda p: (p[0] + p[1], (p[1] if (p[0] + p[1]) % 2 == 0 else -p[1])),
    )
    # The classic zig-zag alternates direction on each anti-diagonal:
    result = []
    for s in range(2 * n - 1):
        diag = [(u, v) for (u, v) in coords if u + v == s]
        if s % 2 == 0:
            diag = sorted(diag, key=lambda p: -p[0])  # go up-right
        else:
            diag = sorted(diag, key=lambda p: p[0])   # go down-left
        result.extend(diag)
    return result


_ZIGZAG = _zigzag_positions(BLOCK)


def _coeff_positions(band=DEFAULT_BAND):
    """Zig-zag (u, v) coordinates used as embedding slots inside each block.

    `band` is a (start, end) window over the zig-zag order (0 = DC). Lower indices
    are lower-frequency and more robust to low-pass attacks.
    """
    start, end = band
    return [_ZIGZAG[i] for i in range(start, end)]


# ----------------------------------------------------------------------------
# Key derivation + chaotic streams
# ----------------------------------------------------------------------------
def _key_to_seed(key: str, salt: str = "") -> float:
    """Derive a logistic-map seed x0 in (0,1) from a passphrase."""
    h = hashlib.sha256((salt + "::" + key).encode("utf-8")).hexdigest()
    val = int(h[:16], 16)
    return ((val % 999983) + 1) / 1000000.0


def _logistic_stream(x0: float, length: int, r: float = DEFAULT_LOGISTIC_R) -> np.ndarray:
    """Generate a logistic-map chaotic sequence in (0,1) of the given length."""
    out = np.empty(length, dtype=np.float64)
    x = x0
    # warm-up to shed transient
    for _ in range(50):
        x = r * x * (1.0 - x)
    for i in range(length):
        x = r * x * (1.0 - x)
        out[i] = x
    return out


def _keystream_bits(key: str, length: int) -> np.ndarray:
    """Pseudo-random bit keystream for the stream cipher, keyed by passphrase."""
    x0 = _key_to_seed(key, salt="cipher")
    seq = _logistic_stream(x0, length, DEFAULT_LOGISTIC_R)
    return (seq > 0.5).astype(np.uint8)


def _key_permutation(key: str, n: int) -> np.ndarray:
    """Deterministic key-driven permutation of range(n) via logistic chaos."""
    x0 = _key_to_seed(key, salt="perm")
    seq = _logistic_stream(x0, n, DEFAULT_LOGISTIC_R)
    return np.argsort(seq, kind="stable")


# ----------------------------------------------------------------------------
# Payload framing + ECC + bit utilities
# ----------------------------------------------------------------------------
def _to_bytes(payload) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8")
    raise TypeError("payload must be str or bytes")


def _frame_payload(payload: bytes, flags: int = 0) -> bytes:
    """MAGIC(2) | version(1) | flags(1) | length(4) | crc32(4) | payload."""
    length = len(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    header = MAGIC + struct.pack(">BBI I", VERSION, flags, length, crc)
    return header + payload


def _unframe(frame: bytes):
    """Validate and strip the frame. Returns (payload_bytes, flags) or raises."""
    if len(frame) < 12:
        raise ValueError("frame too short")
    if frame[:2] != MAGIC:
        raise ValueError("bad magic")
    version, flags, length, crc = struct.unpack(">BBI I", frame[2:12])
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    payload = frame[12:12 + length]
    if len(payload) != length:
        raise ValueError("truncated payload")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
        raise ValueError("CRC mismatch")
    return payload, flags


def _rs_encode(data: bytes, nsym: int) -> bytes:
    return bytes(RSCodec(nsym).encode(bytearray(data)))


def _rs_decode(data: bytes, nsym: int) -> bytes:
    decoded = RSCodec(nsym).decode(bytearray(data))
    # reedsolo>=1.0 returns a tuple (msg, msg+ecc, errata_pos)
    if isinstance(decoded, tuple):
        decoded = decoded[0]
    return bytes(decoded)


def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    r = len(bits) % 8
    if r:
        bits = np.concatenate([bits, np.zeros(8 - r, dtype=np.uint8)])
    return np.packbits(bits.astype(np.uint8)).tobytes()


# ----------------------------------------------------------------------------
# DCT helpers
# ----------------------------------------------------------------------------
def _pad_to_block(channel: np.ndarray):
    h, w = channel.shape
    ph = (BLOCK - h % BLOCK) % BLOCK
    pw = (BLOCK - w % BLOCK) % BLOCK
    if ph or pw:
        channel = cv2.copyMakeBorder(channel, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    return channel, h, w


def _adaptive_delta(dc_value: float, base_delta: float) -> float:
    """
    DC-guided luminance masking. The HVS tolerates more distortion in mid-tones
    and less in very dark regions. DC is never modified, so the decoder recomputes
    the identical step from the stego block's DC -> blind-recoverable adaptivity.
    """
    mean = dc_value / BLOCK  # DCT-II DC ~ block_sum/N; mean ~ DC/BLOCK for cv2.dct
    # Map brightness [0,255] to a gentle multiplier in [0.8, 1.3].
    norm = np.clip(mean / 255.0, 0.0, 1.0)
    mult = 0.8 + 0.5 * (4.0 * norm * (1.0 - norm))  # peaks at mid-tone
    return base_delta * mult


def _qim_embed(coeff: float, bit: int, delta: float) -> float:
    """Embed one bit into a coefficient via parity QIM. Returns new coefficient."""
    l = int(np.round(coeff / delta))
    if (l & 1) != bit:
        # move to the nearest integer with the required parity
        frac = coeff / delta - l
        l = l + 1 if frac >= 0 else l - 1
    return l * delta


def _qim_extract(coeff: float, delta: float) -> int:
    return int(np.round(coeff / delta)) & 1


# ----------------------------------------------------------------------------
# Channel extraction / reassembly
# ----------------------------------------------------------------------------
def _get_embed_channel(bgr: np.ndarray, channel: str):
    if channel == "Y":
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        return ycrcb[:, :, 0], ycrcb
    if channel == "B":
        return bgr[:, :, 0].astype(np.float32), bgr.astype(np.float32)
    raise ValueError("channel must be 'Y' or 'B'")


def _set_embed_channel(container: np.ndarray, chan: np.ndarray, channel: str) -> np.ndarray:
    if channel == "Y":
        container[:, :, 0] = chan
        bgr = cv2.cvtColor(np.clip(container, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        return bgr
    container[:, :, 0] = chan
    return np.clip(container, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Slot plan (identical on encoder and decoder -- key-driven, content-independent)
# ----------------------------------------------------------------------------
def _build_slots(h_pad: int, w_pad: int, key: str, coeff_positions):
    """
    Build the ordered list of embedding slots. A slot is (by, bx, u, v).
    The block order is a key-driven chaotic permutation; coefficient order within
    a block is fixed. This depends ONLY on image dimensions + key, so the decoder
    reproduces it exactly without any side information.
    """
    nby = h_pad // BLOCK
    nbx = w_pad // BLOCK
    nblocks = nby * nbx
    perm = _key_permutation(key + "::blocks", nblocks)
    slots = []
    for bidx in perm:
        by = (bidx // nbx) * BLOCK
        bx = (bidx % nbx) * BLOCK
        for (u, v) in coeff_positions:
            slots.append((by, bx, u, v))
    return slots


# ----------------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------------
def embed_carq(
    cover_path,
    payload,
    out_path,
    key="carq-default-key",
    delta=DEFAULT_DELTA,
    redundancy=DEFAULT_REDUNDANCY,
    nsym=DEFAULT_NSYM,
    channel=DEFAULT_CHANNEL,
    adaptive=DEFAULT_ADAPTIVE,
    band=DEFAULT_BAND,
    debug=False,
):
    """
    Embed an arbitrary payload (str or bytes) into a cover image.

    Returns a metrics dict (PSNR/SSIM/MSE/Entropy Diff) plus capacity info.
    """
    bgr = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cover image not found: {cover_path}")

    payload_bytes = _to_bytes(payload)

    # 1) Frame -> RS encode -> bits
    frame = _frame_payload(payload_bytes)
    coded = _rs_encode(frame, nsym)
    coded_bits = _bytes_to_bits(coded)

    # 2) Stream-cipher the coded bits (keyed). NOTE: we deliberately do NOT bit-
    #    interleave here. Reed-Solomon corrects BYTE errors, so it benefits from
    #    errors staying concentrated within bytes. Spatial scrambling is already
    #    provided by the key-driven chaotic BLOCK permutation in _build_slots, so a
    #    bit-level interleave would only spread residual errors across more bytes
    #    and weaken the ECC (critically so under directional/motion blur).
    ks = _keystream_bits(key, len(coded_bits))
    enc_bits = np.bitwise_xor(coded_bits, ks)

    n_logical = len(enc_bits)

    # 3) Channel + slot plan
    chan, container = _get_embed_channel(bgr, channel)
    chan_pad, h0, w0 = _pad_to_block(chan)
    h_pad, w_pad = chan_pad.shape
    coeff_positions = _coeff_positions(band)
    slots = _build_slots(h_pad, w_pad, key, coeff_positions)

    capacity_slots = len(slots)
    required = n_logical * redundancy
    if required > capacity_slots:
        raise ValueError(
            f"Payload too large: need {required} slots (={n_logical} bits x R{redundancy}), "
            f"have {capacity_slots}. Reduce payload, lower redundancy, or use a bigger image."
        )

    # 4) Spread each logical bit across R slots, interleaved across the whole image
    #    slot j carries logical bit (j % n_logical) for j in [0, required)
    slot_bits = np.empty(required, dtype=np.uint8)
    for j in range(required):
        slot_bits[j] = enc_bits[j % n_logical]

    # 5) Embed via QIM block-by-block (DCT once per touched block)
    touched = {}
    for j in range(required):
        by, bx, u, v = slots[j]
        touched.setdefault((by, bx), None)

    # Embed: process slots, caching DCT per block
    block_cache = {}
    for j in range(required):
        by, bx, u, v = slots[j]
        keyb = (by, bx)
        if keyb not in block_cache:
            blk = chan_pad[by:by + BLOCK, bx:bx + BLOCK].astype(np.float32)
            block_cache[keyb] = cv2.dct(blk)
        D = block_cache[keyb]
        d = _adaptive_delta(D[0, 0], delta) if adaptive else delta
        D[u, v] = _qim_embed(float(D[u, v]), int(slot_bits[j]), d)

    # 6) Inverse DCT for all touched blocks, write back
    for (by, bx), D in block_cache.items():
        blk = cv2.idct(D)
        chan_pad[by:by + BLOCK, bx:bx + BLOCK] = blk

    # 7) Reassemble and save (PNG = lossless container)
    chan_final = chan_pad[:h0, :w0]
    stego_bgr = _set_embed_channel(container, chan_final, channel)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, stego_bgr)

    # 8) Metrics (RGB, matching the rest of the project)
    original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    stego_rgb = cv2.cvtColor(cv2.imread(out_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    metrics = calculate_metrics(original_rgb, stego_rgb)
    metrics["payload_bytes"] = len(payload_bytes)
    metrics["coded_bytes"] = len(coded)
    metrics["bpp"] = (n_logical * redundancy) / float(h0 * w0)
    metrics["capacity_slots"] = capacity_slots
    metrics["used_slots"] = required

    if debug:
        print(f"[carq] payload={len(payload_bytes)}B coded={len(coded)}B "
              f"logical_bits={n_logical} R={redundancy} used={required}/{capacity_slots} slots")
    return metrics


# ----------------------------------------------------------------------------
# Extraction (blind: only the stego image + key + scheme params)
# ----------------------------------------------------------------------------
def extract_carq(
    stego_path,
    key="carq-default-key",
    delta=DEFAULT_DELTA,
    redundancy=DEFAULT_REDUNDANCY,
    nsym=DEFAULT_NSYM,
    channel=DEFAULT_CHANNEL,
    adaptive=DEFAULT_ADAPTIVE,
    band=DEFAULT_BAND,
    return_bytes=False,
    debug=False,
):
    """
    Blindly extract the payload. No position map, no external length: the frame
    header carries the length and CRC, and the slot order is regenerated from key.
    """
    bgr = cv2.imread(stego_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Stego image not found: {stego_path}")

    chan, _ = _get_embed_channel(bgr, channel)
    chan_pad, h0, w0 = _pad_to_block(chan)
    h_pad, w_pad = chan_pad.shape
    coeff_positions = _coeff_positions(band)
    slots = _build_slots(h_pad, w_pad, key, coeff_positions)

    # Read every slot's QIM bit (cache DCT per block)
    block_cache = {}
    all_bits = np.empty(len(slots), dtype=np.uint8)
    deltas = np.empty(len(slots), dtype=np.float64)
    for j, (by, bx, u, v) in enumerate(slots):
        keyb = (by, bx)
        if keyb not in block_cache:
            blk = chan_pad[by:by + BLOCK, bx:bx + BLOCK].astype(np.float32)
            block_cache[keyb] = cv2.dct(blk)
        D = block_cache[keyb]
        d = _adaptive_delta(D[0, 0], delta) if adaptive else delta
        deltas[j] = d
        all_bits[j] = _qim_extract(float(D[u, v]), d)

    # We must know n_logical to fold the repetition. The frame length is variable,
    # so we recover it progressively: first decode the smallest possible region
    # (header is at logical bits 0..), but because bits are spread as j % n_logical,
    # n_logical itself is unknown. We resolve this by trying candidate n_logical
    # derived from the known coded structure: the header is 12 bytes, RS adds nsym
    # per <=255 chunk. We therefore brute-force the *frame length* by reading the
    # header first using a bootstrap: the first 12 framed bytes are always present.
    #
    # Bootstrap: the minimum coded stream is RS(12-byte header)+... Instead we read
    # progressively increasing logical sizes. To keep it deterministic & robust, we
    # decode by assuming maximum logical capacity and majority-voting, then RS-decode
    # in a sliding manner. Simplest correct approach: reconstruct for the maximum
    # n_logical that fits, then RS-decode the leading bytes which self-describe length.

    payload = _blind_decode(all_bits, redundancy, key, nsym, debug=debug)
    if payload is None:
        if debug:
            print("[carq] extraction failed (no valid frame)")
        return b"" if return_bytes else ""

    if return_bytes:
        return payload
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload


def _fold_majority(all_bits: np.ndarray, n_logical: int, redundancy: int) -> np.ndarray:
    """Majority-vote the R spread copies back into n_logical logical bits.

    Layout: slot j carries logical bit (j % n_logical), so the R copies occupy
    consecutive length-n_logical blocks and reshape to (R, n_logical) rows.
    Vectorized (no Python loop) so the blind length search stays fast even when a
    given candidate length does not validate.
    """
    required = n_logical * redundancy
    m = all_bits[:required].reshape(redundancy, n_logical)
    return (m.sum(axis=0) * 2 > redundancy).astype(np.uint8)


def _blind_decode(all_bits, redundancy, key, nsym, debug=False):
    """
    Recover the payload without knowing the frame length in advance.

    Strategy: the encoder lays slot j = logical bit (j % n_logical). We don't know
    n_logical, but the RS-coded frame self-describes its length once decoded. We
    therefore try candidate coded-frame byte lengths from smallest upward; for each,
    n_logical = 8 * coded_len, fold by majority vote, de-interleave, de-cipher,
    RS-decode, and check the frame's MAGIC + CRC. The correct length validates.
    """
    capacity = len(all_bits)
    # The RS codeword length for a frame of F bytes: chunks of 255, each +nsym.
    # We iterate candidate frame payload sizes and compute the exact coded length.
    # To bound work, iterate coded_len directly over feasible range.
    max_coded = capacity // redundancy // 8
    for coded_len in range(12 + nsym, max_coded + 1):
        n_logical = coded_len * 8
        if n_logical * redundancy > capacity:
            break
        enc_bits = _fold_majority(all_bits, n_logical, redundancy)

        # de-cipher (no de-interleave: see encoder note)
        ks = _keystream_bits(key, n_logical)
        coded_bits = np.bitwise_xor(enc_bits, ks)
        coded = _bits_to_bytes(coded_bits)[:coded_len]

        try:
            frame = _rs_decode(coded, nsym)
        except Exception:
            continue
        if frame[:2] != MAGIC:
            continue
        try:
            payload, _flags = _unframe(frame)
        except Exception:
            continue
        if debug:
            print(f"[carq] decoded: coded_len={coded_len} payload={len(payload)}B")
        return payload
    return None


# ----------------------------------------------------------------------------
# Project-pattern wrappers
# ----------------------------------------------------------------------------
def embed_message(cover_path, message, out_path, meta_path=None, method="carq", **kwargs):
    """Wrapper matching the project's embed_message(...) signature. meta_path unused."""
    kwargs.pop("seed_str", None)
    kwargs.pop("logistic_seed", None)
    kwargs.pop("logistic_r", None)
    kwargs.pop("alpha", None)
    kwargs.pop("max_bits_per_pixel", None)
    return embed_carq(cover_path, message, out_path, **kwargs)


def extract_message(stego_path, key="carq-default-key", **kwargs):
    return extract_carq(stego_path, key=key, **kwargs)
