import numpy as np
import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.stego.acme import embed_message as acme_embed, extract_message as acme_extract
from src.stego.lsb_basic import embed_message as lsb_embed, extract_message as lsb_extract
from src.test.attack import add_gaussian_noise, gaussian_blur, median_filter, bilateral_filter, non_local_means_filter, gamma_correction, jpeg_compression, salt_and_pepper_noise, speckle_noise, histogram_equalization, motion_blur, string_ber


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
        "noise_sigma2": lambda x: add_gaussian_noise(x, 2.0),
        "blur": lambda x: gaussian_blur(x, 5, 1.0),
        "median": lambda x: median_filter(x, 5),
        "bilateral": lambda x: bilateral_filter(x, 5, 75, 75),
        "non_local_means": lambda x: non_local_means_filter(x, 10.0),
        "gamma09": lambda x: gamma_correction(x, 0.9),
        "salt_and_pepper": lambda x: salt_and_pepper_noise(x, 0.01),
        "speckle": lambda x: speckle_noise(x),
        "histogram": lambda x: histogram_equalization(x),
        "motion_blur": lambda x: motion_blur(x, 9),
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
            ber = string_ber(secret_message, extracted_str)
            attack_metric[name] = ber
        results[algo] = attack_metric
    return results

def process_directory(input_dir):
    """
    Process all images in input directory and save results to CSV
    """
    # Create outcomes directory if it doesn't exist
    outcomes_dir = Path("outcomes")
    outcomes_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in Path(input_dir).glob('**/*') if f.suffix.lower() in image_extensions]
    
    # Prepare results storage
    all_results = []
    
    # Process each image with progress bar
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Run robustness evaluation
            results = robustness_eval(str(img_path))
            
            # Format results for CSV
            for algo, attack_metrics in results.items():
                row = {
                    'image_name': img_path.name,
                    'algorithm': algo
                }
                row.update(attack_metrics)  # Add all attack metrics
                all_results.append(row)
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    csv_path = outcomes_dir / "robustness_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    # Example assuming acme_embed/acme_extract exist
    process_directory("images/input/googleapi_130_160/temp")

    # for algo, attack_metrics in results.items():
    #     for attack, ber in attack_metrics.items():
    #         print(f"{algo} - {attack}: BER={ber:.4f}")
