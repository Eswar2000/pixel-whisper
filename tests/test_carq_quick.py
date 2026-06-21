"""Quick standalone validation for CARQ-Stego (not part of the paper pipeline).

Checks:
  1. Blind round-trip (no attack) recovers the payload exactly.
  2. BER under a few representative attacks.
  3. PSNR/SSIM of the stego.
"""
import os
import numpy as np
import cv2

import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.stego.carq import embed_carq, extract_carq
from src.test.attack import (
    jpeg_compression, add_gaussian_noise, gaussian_blur, median_filter,
    salt_and_pepper_noise, motion_blur,
)
from src.test.attack import string_ber


def make_synthetic_cover(path, h=512, w=512, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (128 + 80 * np.sin(2 * np.pi * xx / 40) * np.cos(2 * np.pi * yy / 55))
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


def main():
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    cover = make_synthetic_cover("images/_test/cover.png")
    stego = "images/_test/stego.png"
    key = "research-grade-passphrase-2026"

    payload = ("CARQ-Stego research payload: " + ("abcdefghijklmnopqrstuvwxyz0123456789 " * 4)).encode("utf-8")
    log(f"payload size = {len(payload)} bytes")

    metrics = embed_carq(cover, payload, stego, key=key, debug=True)
    log("PSNR={:.2f} SSIM={:.4f} MSE={:.3f} bpp={:.4f} used={}/{}".format(
        metrics["PSNR"], metrics["SSIM"], metrics["MSE"], metrics["bpp"],
        metrics["used_slots"], metrics["capacity_slots"]))

    # 1) Clean round-trip
    rec = extract_carq(stego, key=key, return_bytes=True, debug=True)
    log(f"CLEAN round-trip exact match: {rec == payload}")

    # 2) Attacks
    stego_img = cv2.imread(stego, cv2.IMREAD_COLOR)
    attacks = {
        "jpeg90": lambda x: jpeg_compression(x, 90),
        "jpeg75": lambda x: jpeg_compression(x, 75),
        "noise_sigma2": lambda x: add_gaussian_noise(x, 2.0),
        "blur": lambda x: gaussian_blur(x, 5, 1.0),
        "median": lambda x: median_filter(x, 3),
        "salt_pepper": lambda x: salt_and_pepper_noise(x, 0.01),
        "motion_blur": lambda x: motion_blur(x, 9),
    }
    log("\n--- BER under attack ---")
    for name, fn in attacks.items():
        atk = fn(stego_img)
        atk_path = f"images/_test/atk_{name}.png"
        cv2.imwrite(atk_path, atk)
        rec_b = extract_carq(atk_path, key=key, return_bytes=True)
        try:
            rec_s = rec_b.decode("utf-8")
        except Exception:
            rec_s = ""
        ber = string_ber(payload.decode("utf-8"), rec_s)
        exact = rec_b == payload
        log(f"{name:14s} BER={ber:.4f} exact={exact}")

    with open("images/_test/report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
