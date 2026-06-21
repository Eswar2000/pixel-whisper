"""Physical-layer diagnostic for CARQ-Stego.

Bypasses the high-level decoder. For each (band, delta) and each attack it measures:
  * raw_ber  : per-slot QIM bit error rate (R=1, single copy) vs the known embedded bits
  * fold_ber : bit error rate AFTER majority vote over R copies
  * rs_ok    : whether the full framed payload decodes & CRC-verifies (end-to-end)

This tells us whether QIM physically survives an attack (low raw_ber) and how much
redundancy/ECC is needed -- the data that drives the band/delta/R choice for the paper.
"""
import os
import sys
import numpy as np
import cv2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.stego import carq
from src.test.attack import (
    jpeg_compression, add_gaussian_noise, gaussian_blur, median_filter,
    salt_and_pepper_noise, motion_blur,
)


def make_synthetic_cover(path, h=512, w=512, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 128 + 80 * np.sin(2 * np.pi * xx / 40) * np.cos(2 * np.pi * yy / 55)
    texture = rng.normal(0, 18, size=(h, w))
    chan = np.clip(base + texture, 0, 255)
    img = np.stack([
        np.clip(chan + rng.normal(0, 8, (h, w)), 0, 255),
        np.clip(chan * 0.9 + rng.normal(0, 8, (h, w)), 0, 255),
        np.clip(chan * 1.1 + rng.normal(0, 8, (h, w)), 0, 255),
    ], axis=-1).astype(np.uint8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, img)
    return path


def read_all_slot_bits(stego_bgr, key, band, delta, channel, adaptive):
    chan, _ = carq._get_embed_channel(stego_bgr, channel)
    chan_pad, h0, w0 = carq._pad_to_block(chan)
    h_pad, w_pad = chan_pad.shape
    coeff_positions = carq._coeff_positions(band)
    slots = carq._build_slots(h_pad, w_pad, key, coeff_positions)
    block_cache = {}
    bits = np.empty(len(slots), dtype=np.uint8)
    for j, (by, bx, u, v) in enumerate(slots):
        kb = (by, bx)
        if kb not in block_cache:
            blk = chan_pad[by:by + carq.BLOCK, bx:bx + carq.BLOCK].astype(np.float32)
            block_cache[kb] = cv2.dct(blk)
        D = block_cache[kb]
        d = carq._adaptive_delta(D[0, 0], delta) if adaptive else delta
        bits[j] = carq._qim_extract(float(D[u, v]), d)
    return bits


def fold_majority(bits, n_logical, R):
    required = n_logical * R
    m = bits[:required].reshape(R, n_logical)
    votes = m.sum(axis=0)
    return (votes * 2 > R).astype(np.uint8)


def direct_decode_ok(folded_enc_bits, key, nsym, coded_len, payload):
    """Decode using the KNOWN n_logical (no brute force). Returns 'Y' if exact."""
    n_logical = len(folded_enc_bits)
    interleave = carq._key_permutation(key + "::interleave", n_logical)
    deinter = np.empty_like(folded_enc_bits)
    deinter[interleave] = folded_enc_bits
    ks = carq._keystream_bits(key, n_logical)
    coded_bits = np.bitwise_xor(deinter, ks)
    coded = carq._bits_to_bytes(coded_bits)[:coded_len]
    try:
        frame = carq._rs_decode(coded, nsym)
        rec, _ = carq._unframe(frame)
        return "Y" if rec == payload else "."
    except Exception:
        return "."


def main():
    cover = make_synthetic_cover("images/_test/cover.png")
    key = "research-grade-passphrase-2026"
    payload = ("CARQ-Stego research payload: " + ("abcdefghijklmnopqrstuvwxyz0123456789 " * 4)).encode("utf-8")
    R = 5
    nsym = 16
    channel = "Y"
    adaptive = True

    # Reconstruct the exact embedded enc_bits (encoder-side) so we can compute true BER.
    frame = carq._frame_payload(payload)
    coded = carq._rs_encode(frame, nsym)
    coded_bits = carq._bytes_to_bits(coded)
    ks = carq._keystream_bits(key, len(coded_bits))
    enc_bits = np.bitwise_xor(coded_bits, ks)
    interleave = carq._key_permutation(key + "::interleave", len(enc_bits))
    enc_bits = enc_bits[interleave]
    n_logical = len(enc_bits)

    attacks = {
        "none": lambda x: x,
        "jpeg75": lambda x: jpeg_compression(x, 75),
        "noise2": lambda x: add_gaussian_noise(x, 2.0),
        "blur": lambda x: gaussian_blur(x, 5, 1.0),
        "median3": lambda x: median_filter(x, 3),
        "salt_pep": lambda x: salt_and_pepper_noise(x, 0.01),
        "motion": lambda x: motion_blur(x, 9),
    }

    bands = [(1, 4), (1, 7), (3, 9), (6, 12), (10, 22)]
    deltas = [12.0, 24.0, 40.0, 60.0]

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    for band in bands:
        for delta in deltas:
            stego_path = "images/_test/diag_stego.png"
            m = carq.embed_carq(cover, payload, stego_path, key=key, delta=delta,
                                redundancy=R, nsym=nsym, channel=channel,
                                adaptive=adaptive, band=band)
            psnr = m["PSNR"]
            row = [f"band={band} delta={delta:>5.1f} PSNR={psnr:5.2f}"]
            stego_bgr = cv2.imread(stego_path, cv2.IMREAD_COLOR)
            for aname, afn in attacks.items():
                atk = afn(stego_bgr)
                ap = f"images/_test/diag_atk.png"
                cv2.imwrite(ap, atk)
                atk_bgr = cv2.imread(ap, cv2.IMREAD_COLOR)
                bits = read_all_slot_bits(atk_bgr, key, band, delta, channel, adaptive)
                raw = bits[:n_logical]
                raw_ber = np.mean(raw != enc_bits)
                folded = fold_majority(bits, n_logical, R)
                fold_ber = np.mean(folded != enc_bits)
                ok = direct_decode_ok(folded, key, nsym, len(coded), payload)
                row.append(f"{aname}:{raw_ber:.2f}/{fold_ber:.2f}{ok}")
            log("  ".join(row))
        log("")

    with open("images/_test/diag_report.txt", "w", encoding="utf-8") as f:
        f.write("legend: attack:rawBER/foldBER + (Y=full decode ok)\n")
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
