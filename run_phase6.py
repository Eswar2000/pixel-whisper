import numpy as np
import os
import cv2
from src.stego.acme import embed_message as acme_embed, extract_message as acme_extract
from src.stego.lsb_basic import embed_message as lsb_embed, extract_message as lsb_extract
from src.test.attack import median_filter, bilateral_filter, non_local_means_filter, gamma_correction, jpeg_compression, string_ber


def robustness_eval(cover_img_path):

    results = {}
    img_name = os.path.splitext(os.path.basename(cover_img_path))[0]
    secret_message = "This is your boi Eswar!"
    stego_img_path_acme = f"images/output/attacks/{img_name}_acme_attack.png"
    stego_img_path_lsb = f"images/output/attacks/{img_name}_lsb_attack.png"
    meta_file_acme = f"images/output/meta/{img_name}_meta_attack.json"

    # Embed
    acme_embed(cover_img_path, secret_message, stego_img_path_acme, meta_file_acme, method="acme", seed_str="sussy-chungus", logistic_seed=0.54321, logistic_r=3.99, alpha=0.7, max_bits_per_pixel=2, debug=False)
    lsb_embed(cover_img_path, secret_message, stego_img_path_lsb)
    
    # Attack suite
    attacks = {
        "jpeg90": lambda x: jpeg_compression(x, 100),
        # "jpeg75": lambda x: jpeg_compression(x, 75),
        # "noise_sigma2": lambda x: add_gaussian_noise(x, 2.0),
        # "blur": lambda x: gaussian_blur(x, 5, 1.0),
        "median": lambda x: median_filter(x, 5),
        "bilateral": lambda x: bilateral_filter(x, 5, 75, 75),
        "non_local_means": lambda x: non_local_means_filter(x, 10.0),
        "gamma09": lambda x: gamma_correction(x, 0.9),
    }

    stego_acme, stego_lsb = cv2.imread(stego_img_path_acme), cv2.imread(stego_img_path_lsb)

    for algo in ["acme", "lsb"]:
        attack_metric = {}
        for name, attack in attacks.items():
            attacked = attack(stego_acme if algo == "acme" else stego_lsb)
            attacked_path = f"images/output/attacks/{img_name}_{algo}_attack_{name}.png"
            cv2.imwrite(attacked_path, attacked)
            if algo == "acme":
                extracted_str = acme_extract(attacked_path, meta_file_acme)
            else:
                extracted_str = lsb_extract(attacked_path, len(secret_message))
            print(f"Attack: {name}, Extracted: {extracted_str}")
            ber = string_ber(secret_message, extracted_str)
            attack_metric[name] = ber
        results[algo] = attack_metric
    return results


if __name__ == "__main__":
    # Example assuming acme_embed/acme_extract exist
    results = robustness_eval(cover_img_path = "images/input/googleapi_130_160/img_160.jpg")

    for algo, attack_metrics in results.items():
        for attack, ber in attack_metrics.items():
            print(f"{algo} - {attack}: BER={ber:.4f}")
