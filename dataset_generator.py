"""
    Dataset Generation and Preparation Pipeline
    -------------------------------------------

    This script is the first step in creating a custom dataset for the SnapStock project.
    It automates the process of gathering raw image data from the web.

    Part 1: Image Scraping (This Script)
    1.  Defines a structured list of bakery categories and specific items.
    2.  Iterates through each item to generate a search query (e.g., "photo of fresh croissant").
    3.  Uses the `duckduckgo_search` library to find and collect image URLs related to the query.
    4.  Uses the `requests` library to download the actual image from each URL.
    5.  Saves the downloaded images into a well-organized folder structure, where each
        class of baked good has its own directory (e.g., `./bakery_image_dataset/croissant/`).

    Part 2: Data Annotation and Processing (Roboflow)
    After running this script, the following manual and platform-based steps were taken:
    1.  **Manual Cleaning:** The raw downloaded dataset was manually reviewed to remove any
        irrelevant or low-quality images (e.g., drawings, logos, blurry photos).
    2.  **Upload to Roboflow:** The curated set of clean images was uploaded to a new
        Roboflow project configured for Object Detection.
    3.  **Annotation:** Each image was annotated by manually drawing bounding boxes
        around every baked good and assigning the correct class label.
    4.  **Dataset Generation:** A final version of the dataset was generated with:
        - A balanced 70/20/10 (Train/Validation/Test) split.
        - Preprocessing: All images were resized to a standard 640x640 pixels.
        - Augmentation: The training set was expanded with augmentations like flips,
          rotations, and brightness adjustments to create a more robust model.
    5.  **Export:** The final, ready-to-use dataset was exported in the YOLOv8 format,
        which included the structured image/label folders and the `data.yaml` file.
"""
import os
import requests
import time
from duckduckgo_search import DDGS

BAKED_GOODS_CATEGORIES = {
	"breads": ["baguette", "sourdough bread", "ciabatta bread"],
	"pastries": ["croissant", "pain au chocolat", "muffin", "donut"],
	"cakes_cupcakes": ["cheesecake", "red velvet cake", "cupcake"],
	"tarts_pies": ["fruit tart", "apple pie", "quiche"],
	"cookies_desserts": ["macaron", "eclair", "chocolate chip cookie"]
}

BASE_DOWNLOAD_PATH = "bakery_image_dataset"
IMAGES_PER_ITEM = 150

# It's still good practice to identify your script with headers
HEADERS = {
	'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def download_images_from_urls(urls, output_folder, item_name):
	"""Downloads images from a list of URLs into a specified folder."""
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	img_count = 0
	for i, url in enumerate(urls):
		try:
			response = requests.get(url, headers=HEADERS, timeout=15)
			response.raise_for_status()

			file_path = os.path.join(output_folder, f"{item_name}_{i + 1}.jpg")
			with open(file_path, 'wb') as f:
				f.write(response.content)
			img_count += 1

		except Exception as e:
			pass  # Silently ignore download errors
	return img_count


# --- Main Loop ---
if not os.path.exists(BASE_DOWNLOAD_PATH):
	os.makedirs(BASE_DOWNLOAD_PATH)

# Use the DDGS context manager from the new library
with DDGS() as ddgs:
	for category, items in BAKED_GOODS_CATEGORIES.items():
		for item in items:
			item_folder_name = item.replace(" ", "_")
			output_directory = os.path.join(BASE_DOWNLOAD_PATH, item_folder_name)

			print(f"\n--- Processing item: {item.upper()} ---")

			# This is the new, simplified way to get the image URLs
			search_results = ddgs.images(
				keywords=f"photo of fresh {item}",
				max_results=IMAGES_PER_ITEM
			)

			# The library returns a generator, so we extract the 'image' URL from each result
			urls = [r['image'] for r in search_results]

			if urls:
				print(f"Found {len(urls)} image URLs. Starting download...")
				downloaded_count = download_images_from_urls(urls, output_directory, item_folder_name)
				print(f"Successfully downloaded {downloaded_count} images.")
			else:
				print("No URLs found for this item.")

			time.sleep(2)  # Pause between different items

print(f"\n--- Download process complete! ---")