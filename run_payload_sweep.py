# run_payload_sweep.py
"""Capacity-imperceptibility frontier experiment.

For every cover image and every algorithm, embed payloads of increasing size and
record PSNR / SSIM / MSE versus payload (bytes and bits-per-pixel, bpp). Each
algorithm is swept up to its own capacity (the embed call raises once the payload
no longer fits, at which point we stop growing the payload for that algorithm on
that image).

This produces the capacity-vs-imperceptibility curves that a paper needs:
  * the LSB family sustains high PSNR but (separately) is fragile to attacks;
  * CARQ's curve sits lower AND terminates early -- the honest cost of robustness;
  * CARQ is shown at R=5 (robust) and R=1 (high-capacity) to expose its tunable
    capacity<->robustness knob.

Outputs:
  outcomes/payload_sweep.csv
  images/plot/payload_psnr_vs_bpp.png
  images/plot/payload_ssim_vs_bpp.png

Usage:
  python run_payload_sweep.py [input_dir] [max_images]
"""
import os
import sys
import random
import string

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Algorithm embed functions (heterogeneous APIs normalized via adapters) ---
from src.stego.lsb_basic import embed_message as _lsb_embed
from src.stego.chaos_dynamic_lsb import embed_message_dynamic as _dyn_embed
from src.stego.chaos_content_aware_lsb import embed_message_content_aware as _ca_embed
from src.stego.chaos_compensation import embed_message as _chcomp_embed
from src.stego.lsb_compensation import embed_message as _lsbcomp_embed
from src.stego.meta_compensation import embed_message as _meta_embed
from src.stego.acme import embed_message as _acme_embed
from src.stego.carq import embed_carq as _carq_embed

OUT_DIR = os.path.join("images", "output", "sweep")
META_DIR = os.path.join(OUT_DIR, "meta")
PLOT_DIR = os.path.join("images", "plot")
CSV_PATH = os.path.join("outcomes", "payload_sweep.csv")

# Payload sizes in bytes (ascending). The sweep stops per (image, algo) once a
# size no longer fits, so listing large sizes is harmless.
PAYLOAD_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

_SEED_STR = "sussy-chungus"
_CARQ_KEY = "carq-research-key-2026"


# ---------------------------------------------------------------------------
# Adapters: each takes (cover, message, out_path, meta_path) -> metrics dict
# ---------------------------------------------------------------------------
def _a_lsb(cover, msg, out, meta):
    return _lsb_embed(cover, msg, out)

def _a_dyn(cover, msg, out, meta):
    return _dyn_embed(cover, msg, out, meta)

def _a_ca(cover, msg, out, meta):
    return _ca_embed(cover, msg, out, meta)

def _a_chcomp(cover, msg, out, meta):
    return _chcomp_embed(cover, msg, out, _SEED_STR)

def _a_lsbcomp(cover, msg, out, meta):
    # NOTE: this embed signature is (cover, output, message, channel)
    return _lsbcomp_embed(cover, out, msg, channel_to_use=0)

def _a_meta(cover, msg, out, meta):
    return _meta_embed(cover, msg, out, meta, method="super_model",
                       seed_str=_SEED_STR, logistic_seed=0.54321,
                       logistic_r=3.99, alpha=0.7)

def _a_acme(cover, msg, out, meta):
    return _acme_embed(cover, msg, out, meta, method="acme", seed_str=_SEED_STR,
                       logistic_seed=0.54321, logistic_r=3.99, alpha=0.7,
                       max_bits_per_pixel=2)

def _a_carq_r5(cover, msg, out, meta):
    return _carq_embed(cover, msg, out, key=_CARQ_KEY, redundancy=5)

def _a_carq_r1(cover, msg, out, meta):
    return _carq_embed(cover, msg, out, key=_CARQ_KEY, redundancy=1)


# (label, adapter) -- order controls plot legend / CSV grouping
ALGORITHMS = [
    ("LSB", _a_lsb),
    ("Chaotic Dynamic LSB", _a_dyn),
    ("Content-Aware LSB", _a_ca),
    ("Chaos Compensation", _a_chcomp),
    ("LSB Compensation", _a_lsbcomp),
    ("Meta Compensation", _a_meta),
    ("ACME", _a_acme),
    ("CARQ (R5, robust)", _a_carq_r5),
    ("CARQ (R1, capacity)", _a_carq_r1),
]


def make_payload(n_bytes, seed=12345):
    """Deterministic ASCII payload of exactly n_bytes characters (latin-1 safe)."""
    rng = random.Random(seed * 100003 + n_bytes)
    alphabet = string.ascii_letters + string.digits + " .,!?"
    return "".join(rng.choice(alphabet) for _ in range(n_bytes))


def _safe_label(label):
    return label.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")


def sweep_image(cover_path, rows):
    img = cv2.imread(cover_path)
    if img is None:
        print(f"[!] Could not read {cover_path}, skipping")
        return
    h, w = img.shape[:2]
    n_pixels = h * w
    img_name = os.path.splitext(os.path.basename(cover_path))[0]

    for label, adapter in ALGORITHMS:
        slug = _safe_label(label)
        for size in PAYLOAD_SIZES:
            msg = make_payload(size)
            out = os.path.join(OUT_DIR, f"{img_name}_{slug}_{size}.png")
            meta = os.path.join(META_DIR, f"{img_name}_{slug}_{size}.json")
            try:
                m = adapter(cover_path, msg, out, meta)
            except Exception as e:
                # Capacity reached (or unsupported size) -> stop growing for this algo
                print(f"    {label}: stopped at {size}B ({type(e).__name__})")
                break
            bpp = (size * 8) / n_pixels
            rows.append({
                "image": os.path.basename(cover_path),
                "algorithm": label,
                "payload_bytes": size,
                "bpp": bpp,
                "PSNR": m.get("PSNR"),
                "SSIM": m.get("SSIM"),
                "MSE": m.get("MSE"),
            })
        print(f"  [done] {label}")


def plot_frontier(df, ycol, ylabel, out_path):
    plt.figure(figsize=(8, 5))
    for label, _ in ALGORITHMS:
        sub = df[df["algorithm"] == label]
        if sub.empty:
            continue
        agg = sub.groupby("payload_bytes").agg(bpp=("bpp", "mean"),
                                               y=(ycol, "mean")).reset_index()
        agg = agg.sort_values("bpp")
        plt.plot(agg["bpp"], agg["y"], marker="o", markersize=4, label=label)
    plt.xscale("log")
    plt.xlabel("Payload (bits per pixel, log scale)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs payload (capacity-imperceptibility frontier)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[+] Saved plot: {out_path}")


def main(input_dir="images/input", max_images=5):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    covers = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))
              if f.lower().endswith(exts)]
    if not covers:
        print(f"[!] No images found in {input_dir}")
        return
    covers = covers[:max_images]

    rows = []
    for i, cover in enumerate(covers, 1):
        print(f"\n[*] ({i}/{len(covers)}) Sweeping {os.path.basename(cover)}")
        sweep_image(cover, rows)

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n[+] Saved sweep CSV: {CSV_PATH}  ({len(df)} rows)")

    if not df.empty:
        plot_frontier(df, "PSNR", "PSNR (dB)",
                      os.path.join(PLOT_DIR, "payload_psnr_vs_bpp.png"))
        plot_frontier(df, "SSIM", "SSIM",
                      os.path.join(PLOT_DIR, "payload_ssim_vs_bpp.png"))


if __name__ == "__main__":
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "images/input"
    max_imgs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(in_dir, max_imgs)
