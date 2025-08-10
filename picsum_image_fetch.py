import os
import requests

DATASOURCE_URL = "https://picsum.photos/v2/list"

def fetch_image_list(page=1, limit=100):
    """Fetch image metadata from Picsum."""
    resp = requests.get(f"{DATASOURCE_URL}?page={page}&limit={limit}")
    resp.raise_for_status()
    return resp.json()

def download_images(offset, count, output_dir):
    """Download images starting at offset for count."""
    os.makedirs(output_dir, exist_ok=True)
    
    page = (offset // 100) + 1
    start_index = offset % 100

    images = fetch_image_list(page=page, limit=100)
    subset = images[start_index:start_index + count]

    for idx, img in enumerate(subset, start=1):
        img_url = img["download_url"]
        filename = os.path.join(output_dir, f"image_{offset+idx}.jpg")
        print(f"Downloading {filename} from {img_url}")
        try:
            img_data = requests.get(img_url, timeout=30)
            img_data.raise_for_status()
            with open(filename, "wb") as f:
                f.write(img_data.content)
            print(f"Downloaded: {filename}")
        except requests.exceptions.Timeout:
            print(f"Skipped (timeout): {img_url}")
        except requests.exceptions.RequestException as e:
            print(f"Skipped (error): {img_url} -> {e}")

if __name__ == "__main__":
    download_images(offset=0, count=30, output_dir="images/input/picsum_0_30")
