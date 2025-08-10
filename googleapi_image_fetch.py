import os
import csv
import requests

def download_images(offset, count, dest_dir, url_column='url', timeout=30):
    csv_path = "images/input/googleapi/image_list.csv"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    total_rows = len(rows)
    end_index = min(offset + count, total_rows)

    print(f"Downloading images from index {offset} to {end_index - 1} (total {end_index - offset})")

    for i in range(offset, end_index):
        row = rows[i]
        if url_column not in row:
            print(f"URL column '{url_column}' not found in CSV")
            break

        url = row[url_column]
        ext = os.path.splitext(url.split("?")[0])[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            ext = '.jpg'  # default extension

        save_path = os.path.join(dest_dir, f"img_{i + 1}{ext}")

        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {save_path}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    # hardcoded parameters
    offset = 0
    count = 30
    dest_dir = "images/input/googleapi_0_30"
    url_column = "OriginalURL"

    download_images(offset, count, dest_dir, url_column)
