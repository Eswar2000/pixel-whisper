# run_carq.py
"""Runner for CARQ-Stego (Chaotic Adaptive Robust QIM steganography).

CARQ is the research-grade successor to ACME. Unlike the earlier phases it is
BLIND: extraction needs only the stego image and the passphrase key -- there is
NO per-image meta / pixel-position file. This runner embeds a parameterized
message, saves the stego image as ``<img>_carq.png`` so it slots into
master_runner alongside the other algorithms, verifies blind extraction, and
returns the standard metrics dict (PSNR / SSIM / MSE / Entropy Diff + capacity).
"""
import os
import sys

from src.stego.carq import embed_carq, extract_carq

DEFAULT_MESSAGE = "The quick brown fox jumps over the lazy dog"
DEFAULT_KEY = "carq-research-key-2026"

# Phase label -> drives the stego filename (<img>_carq.png) and the CSV "Phase"
# column, keeping the master pipeline + plotters consistent with the other algos.
PHASE_LABEL = "carq"

# CARQ is transform-domain and capacity-hungry, so small covers may not hold the
# payload at the default redundancy. We degrade gracefully (R is normally a fixed
# key parameter; we only relax it to fit heterogeneous cover sizes in the
# comparison pipeline). Extraction always uses the matching R.
_REDUNDANCY_FALLBACKS = (5, 3, 1)


def run_carq(cover_img_path, message=DEFAULT_MESSAGE, key=DEFAULT_KEY, debug=False):
    output_dir = os.path.join("images", "output")
    os.makedirs(output_dir, exist_ok=True)

    img_name = os.path.splitext(os.path.basename(cover_img_path))[0]
    stego_img_path = os.path.join(output_dir, f"{img_name}_{PHASE_LABEL}.png")
    secret_message = message

    metrics = None
    used_r = None
    last_err = None
    for redundancy in _REDUNDANCY_FALLBACKS:
        try:
            metrics = embed_carq(cover_img_path, secret_message, stego_img_path,
                                 key=key, redundancy=redundancy, debug=debug)
            used_r = redundancy
            break
        except ValueError as e:
            last_err = e
    if metrics is None:
        raise ValueError(
            f"CARQ could not embed the message into {cover_img_path} even at "
            f"R={_REDUNDANCY_FALLBACKS[-1]} (cover too small): {last_err}"
        )

    if debug:
        print(f"[carq] embedded with redundancy R={used_r}")
        print("[+] Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")

    # Blind extraction -- key only, no meta file. Uses the same R that was embedded.
    extracted = extract_carq(stego_img_path, key=key, redundancy=used_r)
    if debug:
        print("[*] Extracting message (blind)...")
        print("[+] Extracted message:", extracted)
        if extracted != secret_message:
            print("[!] WARNING: blind extraction did not match the original message")
    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_carq.py <cover_image_path> [secret_message]")
        sys.exit(1)
    cover_img_path = sys.argv[1]
    msg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MESSAGE
    run_carq(cover_img_path, message=msg, debug=True)
